/** Conservative exposure update from native-resolution, ROI-only statistics. */
function nextExposure(control, meter) {
  if (!meter.count || !Number.isFinite(meter.peak) || !Number.isFinite(meter.percentile)) throw new Error('Invalid exposure statistics')
  // A percentile sets the working brightness; the peak independently prevents
  // even a small saturated region from hiding below that percentile.
  let ratio = Math.min(.9 / Math.max(meter.percentile, .001), .98 / Math.max(meter.peak, .001))
  if (meter.clipped > 0) ratio = Math.min(ratio, .8)
  if (ratio >= .98 && ratio <= 1.02 && meter.clipped === 0) return control.value
  ratio = Math.max(.5, Math.min(1.2, ratio < 1 ? ratio : 1 + .5 * (ratio - 1)))
  return Math.max(control.min, Math.min(control.max, control.value * ratio))
}
module.exports = { nextExposure }
