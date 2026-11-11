"""Classical reference fitters and a registry for user-written measurement modules.

These diagnostic fits are not standardized minimum-zone GD&T evaluations.
All fitters consume original contour points and report residuals/selection failures.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def selected_points(contours,feature):
    xy=np.asarray(contours['actual_xy_mm'],float);offsets=contours['ring_offsets'];indices=np.array(feature['raw_edge_indices'],int)
    if len(np.unique(indices))!=len(indices) or (indices<0).any() or (indices>=len(xy)).any():raise ValueError('Invalid raw edge selection')
    end=indices+1
    for lo,hi in zip(offsets[:-1],offsets[1:]):end[(indices>=lo)&(indices<hi)&(end==hi)]=lo
    return xy[np.unique(np.r_[indices,end])]


def line_fit(points):
    if len(points)<4:raise ValueError('Insufficient line support')
    center=points.mean(0);_,_,vt=np.linalg.svd(points-center,full_matrices=False)
    axis=vt[0];normal=np.array([-axis[1],axis[0]]);residual=(points-center)@normal
    u=(points-center)@axis
    return {'operator':'total_least_squares_line','center':center.tolist(),'direction':axis.tolist(),
            'projected_length':float(np.ptp(u)),'rms_profile_residual':float(np.sqrt(np.mean(residual**2))),
            'peak_to_valley_profile_residual':float(np.ptp(residual)),'point_count':len(points)}


def circle_fit(points):
    if len(points)<8:raise ValueError('Insufficient circle support')
    c=points.mean(0);r=np.linalg.norm(points-c,axis=1).mean()
    fit=least_squares(lambda z:np.linalg.norm(points-z[:2],axis=1)-z[2],np.r_[c,r],bounds=([-np.inf,-np.inf,1e-9],[np.inf]*3))
    if not fit.success:raise ValueError('Circle fitting failed')
    residual=fit.fun
    angles=np.sort(np.arctan2(points[:,1]-fit.x[1],points[:,0]-fit.x[0]));gap=np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
    return {'operator':'least_squares_circle','center':fit.x[:2].tolist(),'diameter':float(2*fit.x[2]),
            'rms_profile_residual':float(np.sqrt(np.mean(residual**2))),
            'radial_peak_to_valley':float(np.ptp(residual)),'angular_coverage_radians':float(2*np.pi-gap),
            'point_count':len(points),'standardized_roundness':False}


def slot_fit(points):
    if len(points)<24:raise ValueError('Insufficient slot support')
    center=points.mean(0);_,_,vt=np.linalg.svd(points-center,full_matrices=False);axis=vt[0]
    angle=np.arctan2(axis[1],axis[0]);local=(points-center)@vt.T
    r=max(np.ptp(local[:,1])/2,1e-5);a=max(np.ptp(local[:,0])/2-r,1e-5)
    def residual(z):
        c=z[:2];t=z[2];a,r=z[3:];u=(points-c)@np.array([np.cos(t),np.sin(t)]);v=(points-c)@np.array([-np.sin(t),np.cos(t)])
        return np.sqrt(np.maximum(np.abs(u)-a,0)**2+v*v)-r
    fitted=least_squares(residual,[*center,angle,a,r],bounds=([-np.inf]*3+[0,1e-9],[np.inf]*5),max_nfev=200)
    if not fitted.success:raise ValueError('Slot fitting failed')
    z=fitted.x
    return {'operator':'least_squares_stadium_profile','center':z[:2].tolist(),'axis_angle_radians':float(z[2]),
            'width':float(2*z[4]),'overall_length':float(2*(z[3]+z[4])),
            'rms_profile_residual':float(np.sqrt(np.mean(fitted.fun**2))),
            'peak_to_valley_profile_residual':float(np.ptp(fitted.fun)),'point_count':len(points)}


def thread_fit(contours,feature):
    points=selected_points(contours,feature);center=points.mean(0)
    xy=np.asarray(contours['actual_xy_mm']);spans=[];results=[]
    for span in feature['contour_spans']:
        ri=span['ring_index'];lo,hi=contours['ring_offsets'][ri:ri+2];n=hi-lo
        indices=lo+(span['start_edge']+np.arange(span['edge_count']+1))%n
        spans.append(xy[indices])
    if not spans:raise ValueError('No observed thread spans')
    # A short threaded zone can be wider across its diameter than along its axis.
    # Estimate direction on one continuous side; pooling both sides can rotate PCA 90 degrees.
    support=max(spans,key=lambda q:np.ptp(q,axis=0).max())
    _,_,vt=np.linalg.svd(support-support.mean(0),full_matrices=False);axis=vt[0];normal=vt[1]
    for q in spans:
        u=(q-center)@axis;v=(q-center)@normal
        # A disconnected side is a monotone profile. Reject closed/multivalued regions.
        order=np.argsort(u);u=u[order];v=v[order];unique=np.r_[True,np.diff(u)>1e-8];u=u[unique];v=v[unique]
        if len(u)<20:continue
        grid=np.linspace(u[0],u[-1],max(256,len(u)));profile=np.interp(grid,u,v)
        trend=np.polyval(np.polyfit(grid,profile,1),grid);signal=profile-trend
        for sign in (1,-1):
            peaks,_=find_peaks(sign*signal,prominence=max(.3*np.ptp(signal),1e-8))
            if len(peaks)<4:continue
            delta=np.diff(grid[peaks]);pitch=float(np.median(delta));cv=float(np.std(delta)/max(pitch,1e-9))
            if cv<.25:results.append({'pitch':pitch,'spacing_cv':cv,'peak_count':len(peaks)})
    if not results:raise ValueError('No consistent repeated profile with at least four peaks')
    pitch=float(np.median([r['pitch'] for r in results]))
    return {'operator':'visible_profile_peak_pitch','pitch':pitch,'side_estimates':results,
            'axis_direction':axis.tolist(),'point_count':len(points),'is_thread_lead':False}

class MeasurementRouter:
    """Register a callable(contours, feature) for a semantic route or override defaults."""
    def __init__(self):
        self.modules={'line':lambda c,f:line_fit(selected_points(c,f)),
                      'circle':lambda c,f:circle_fit(selected_points(c,f)),
                      'arc':lambda c,f:circle_fit(selected_points(c,f)),
                      'slot':lambda c,f:slot_fit(selected_points(c,f)), 'thread_pitch':thread_fit}
    def register(self,route,module):self.modules[route]=module
    def measure(self,contours,feature):
        base={'feature_id':feature['id'],'semantic':feature['semantic'],'units':feature['units']}
        if feature['status']!='accepted':return {**base,'status':'abstain','reason':'Feature selection confidence below threshold'}
        route=feature['measurement_route']
        if route not in self.modules:return {**base,'status':'unsupported','reason':f'Register a {route} measurement module'}
        if route in ('circle','slot') and not any(s['closed_ring'] for s in feature['contour_spans']):
            return {**base,'status':'abstain','reason':'Closed feature boundary is incomplete'}
        try:
            values=self.modules[route](contours,feature)
            return {**base,'status':'measured','selection_confidence':feature['confidence'],'values':values}
        except (ValueError,np.linalg.LinAlgError) as error:return {**base,'status':'fit_failed','reason':str(error)}
