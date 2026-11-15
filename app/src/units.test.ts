import { describe, expect, it } from 'vitest'
import { formattedValue } from './annotationLayout'
import { displayLength, formatLength, pixelsPerMmFrom, pixelsPerUnit, type LengthUnit, type UnitScale } from './units'

const scale = (lengthUnit: LengthUnit, pixelsPerMm: number | null = 200): UnitScale => ({ lengthUnit, pixelsPerMm })

describe('length units', () => {
  it('converts pixel lengths through the stored px/mm factor', () => {
    expect(displayLength(500, 'px', scale('mm')).value).toBeCloseTo(2.5, 12)
    expect(displayLength(500, 'px', scale('um')).value).toBeCloseTo(2500, 9)
    expect(displayLength(508, 'px', scale('in')).value).toBeCloseTo(0.1, 12)
    expect(formatLength(500, 'px', scale('mm'))).toBe('2.5000 mm')
    expect(formatLength(500, 'px', scale('um'))).toBe('2500.00 µm')
  })

  it('shows pixels when px is selected or no scale is set', () => {
    expect(formatLength(500, 'px', scale('px'))).toBe('500.000 px')
    expect(formatLength(500, 'px', scale('mm', null))).toBe('500.000 px')
  })

  it('leaves non-length units alone', () => {
    expect(displayLength(45, 'deg', scale('mm'))).toMatchObject({ value: 45, units: 'deg' })
  })

  it('switching the entry unit back and forth never changes the stored factor', () => {
    const stored = 192.3456789
    let shown = pixelsPerUnit(stored, 'in')
    for (let i = 0; i < 1001; i++) shown = pixelsPerUnit(stored, i % 2 ? 'um' : 'in')
    // Display is always derived from the stored value, so it is exact after any number of switches.
    expect(shown).toBe(pixelsPerUnit(stored, 'in'))
    expect(pixelsPerMmFrom(pixelsPerUnit(stored, 'in'), 'in')).toBeCloseTo(stored, 12)
  })

  it('uses the decimal places chosen for the shown unit', () => {
    const custom = { ...scale('mm'), lengthDecimals: { px: 1, mm: 2, um: 0, in: 6 } }
    expect(formatLength(500, 'px', custom)).toBe('2.50 mm')
    expect(formatLength(500, 'px', { ...custom, lengthUnit: 'um' })).toBe('2500 µm')
    expect(formatLength(500, 'px', { ...custom, pixelsPerMm: null })).toBe('500.0 px')
    expect(displayLength(45, 'deg', custom).decimals).toBe(3)
  })

  it('converts on-image labels', () => {
    const m = { id: 'r', operator: 'radius', feature_refs: [], status: 'measured', value: 1000, units: 'px' }
    expect(formattedValue(m, scale('mm'))).toBe('R5.0000 mm')
    expect(formattedValue(m, scale('px'))).toBe('R1000.000 px')
  })
})
