/** Temporal feedback over the native bridge's relative sharpness samples. */
class FocusTracker {
  constructor() { this.reset() }
  reset() { this.score = null; this.peak = null; this.time = null; this.sequence = null }
  update(sample, now) {
    if (sample.sequence === this.sequence) return null
    this.sequence = sample.sequence
    if (sample.status !== 'ok' || !Number.isFinite(sample.score) || sample.score <= 0) {
      this.score = null; this.time = null
      return { score: null, peak: this.peak, status: ['too-small', 'unavailable'].includes(sample.status) ? sample.status : 'low-detail', updatedAt: now }
    }
    const alpha = this.time == null ? 1 : 1 - Math.exp(-Math.max(0, now-this.time)/200)
    this.score = this.score == null ? sample.score : this.score + alpha*(sample.score-this.score)
    this.peak = Math.max(this.peak || 0, this.score)
    this.time = now
    return { score: this.score, peak: this.peak, status: 'ok', updatedAt: now }
  }
}
module.exports = { FocusTracker }
