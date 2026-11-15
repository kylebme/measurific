import { create } from 'zustand'

/** Display units for lengths. Measurements are always computed in pixels. */
export type LengthUnit = 'px' | 'mm' | 'um' | 'in'

export const lengthUnitLabels: Record<LengthUnit, string> = { px: 'px', mm: 'mm', um: 'µm', in: 'in' }
export type PhysicalUnit = Exclude<LengthUnit, 'px'>
const mmPerUnit: Record<PhysicalUnit, number> = { mm: 1, um: 0.001, in: 25.4 }
export type LengthDecimals = Record<LengthUnit, number>
export const defaultLengthDecimals: LengthDecimals = { px: 3, mm: 4, um: 2, in: 5 }
export const maxLengthDecimals = 8

/**
 * The pixel scale is stored only as pixels per millimetre. Every displayed
 * value is converted once, from pixels, so switching units never compounds
 * rounding error.
 */
export interface UnitScale { lengthUnit: LengthUnit; pixelsPerMm: number | null; lengthDecimals?: Partial<LengthDecimals> }

export const useUnitScale = create<UnitScale & { setUnitScale: (scale: UnitScale) => void }>((set) => ({
  lengthUnit: 'mm', pixelsPerMm: null, lengthDecimals: defaultLengthDecimals,
  setUnitScale: ({ lengthUnit, pixelsPerMm, lengthDecimals }) => set({ lengthUnit, pixelsPerMm, lengthDecimals: { ...defaultLengthDecimals, ...lengthDecimals } }),
}))

export const currentUnitScale = (): UnitScale => {
  const { lengthUnit, pixelsPerMm, lengthDecimals } = useUnitScale.getState()
  return { lengthUnit, pixelsPerMm, lengthDecimals }
}

export const hasPixelScale = (scale: UnitScale) => scale.pixelsPerMm != null && scale.pixelsPerMm > 0

/** Pixels per one of `unit`, for showing and editing the stored px/mm factor. */
export const pixelsPerUnit = (pixelsPerMm: number, unit: PhysicalUnit) => pixelsPerMm * mmPerUnit[unit]
export const pixelsPerMmFrom = (perUnit: number, unit: PhysicalUnit) => perUnit / mmPerUnit[unit]
/** The unit the scale is entered in: the selected one, or mm while showing pixels. */
export const scaleEntryUnit = (unit: LengthUnit): PhysicalUnit => unit === 'px' ? 'mm' : unit

export const decimalsFor = (unit: LengthUnit, scale: UnitScale) => scale.lengthDecimals?.[unit] ?? defaultLengthDecimals[unit]

/** The unit lengths are actually shown in: pixels until a scale is set. */
export const shownLengthUnit = (scale: UnitScale): LengthUnit => hasPixelScale(scale) ? scale.lengthUnit : 'px'

/**
 * Converts a value reported in `units`; only pixel lengths are rescaled.
 * `isLength` values use the user's decimal places for the shown unit;
 * other values (angles, counts) keep `decimals` as a suggestion only.
 */
export function displayLength(value: number, units: string | undefined, scale: UnitScale = currentUnitScale()) {
  if (units !== 'px') return { value, units, decimals: 3, isLength: false }
  const unit = shownLengthUnit(scale)
  return {
    value: unit === 'px' ? value : value / scale.pixelsPerMm! / mmPerUnit[unit],
    units: lengthUnitLabels[unit],
    decimals: decimalsFor(unit, scale),
    isLength: true,
  }
}

/** Number and unit label, e.g. "1.2345 mm". */
export function formatLength(value: number | null | undefined, units: string | undefined, scale: UnitScale = currentUnitScale()) {
  if (value == null || !Number.isFinite(value)) return '—'
  const shown = displayLength(value, units, scale)
  return `${shown.value.toFixed(shown.decimals)}${shown.units ? ` ${shown.units}` : ''}`
}
