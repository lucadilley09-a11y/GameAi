// What project tests can use (gameai/repair/debug_runner.mjs provides it), so the type checker knows these names.
declare function test(name: string, fn: () => unknown): void;
declare function assert(condition: unknown, message?: string): asserts condition;
interface Matchers {
  toBe(expected: unknown): true; toEqual(expected: unknown): true; toBeTruthy(): true; toBeFalsy(): true;
  toBeGreaterThan(n: number): true; toBeLessThan(n: number): true; toBeCloseTo(n: number, digits?: number): true;
  toContain(item: unknown): true; toThrow(): true;
}
declare function expect(actual: unknown): Matchers & { not: Matchers };
