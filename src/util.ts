import binaryen from "binaryen";

type Bool = 0 | 1;

type Int = number;

type Pointer<T> = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L85-L89 */
export type BinaryenIndex = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L132-L134 */
export type BinaryenPackedType = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L140-L142 */
export type BinaryenHeapType = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3507-L3514 */
type TypeBuilderRef = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3516 */
export type TypeBuilderErrorReason = number;

interface Internal {
  _malloc(size: number): Pointer<any>;
  __i32_store8(pointer: Pointer<number>, value: number): void;
  __i32_store(pointer: Pointer<number>, value: number): void;
  __i32_load(pointer: Pointer<number>): number;
  _free(pointer: Pointer<any>): void;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L136 */
  _BinaryenPackedTypeNotPacked(): BinaryenPackedType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L188-L189 */
  _BinaryenTypeFromHeapType(
    heapType: BinaryenHeapType,
    nullable: Bool,
  ): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3532-L3534 */
  _TypeBuilderCreate(size: BinaryenIndex): TypeBuilderRef;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3546-L3552 */
  _TypeBuilderSetStructType(
    builder: TypeBuilderRef,
    index: BinaryenIndex,
    fieldTypes: Pointer<binaryen.Type>,
    fieldPackedTypes: Pointer<BinaryenPackedType>,
    fieldMutables: Pointer<Bool>,
    numFields: Int,
  ): void;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3583-L3589 */
  _TypeBuilderBuildAndDispose(
    builder: TypeBuilderRef,
    heapTypes: Pointer<BinaryenHeapType>,
    errorIndex: Pointer<BinaryenIndex>,
    errorReason: Pointer<TypeBuilderErrorReason>,
  ): Bool;
}

const internal: Internal = binaryen as any;

const toBool = (x: boolean): Bool => (x ? 1 : 0);

const withAlloc = <T>(size: number, f: (pointer: Pointer<any>) => T): T => {
  const pointer = internal._malloc(size);
  try {
    return f(pointer);
  } finally {
    internal._free(pointer);
  }
};

export const packedTypeNotPacked: BinaryenPackedType =
  internal._BinaryenPackedTypeNotPacked();

export const typeFromHeapType = (
  heapType: BinaryenHeapType,
  nullable: boolean,
): binaryen.Type =>
  internal._BinaryenTypeFromHeapType(heapType, toBool(nullable));

export interface Field {
  type: binaryen.Type;
  packedType: BinaryenPackedType;
  mutable: boolean;
}

export class TypeBuilder {
  constructor(private ref: TypeBuilderRef) {}

  setStructType(index: BinaryenIndex, fields: Field[]): void {
    withAlloc(9 * fields.length, (pointer) => {
      const types = pointer;
      const packedTypes = pointer + 4 * fields.length;
      const mutables = pointer + 8 * fields.length;
      for (let i = 0; i < fields.length; ++i) {
        const { type, packedType, mutable } = fields[i];
        internal.__i32_store(types + 4 * i, type);
        internal.__i32_store(packedTypes + 4 * i, packedType);
        internal.__i32_store8(mutables + i, toBool(mutable));
      }
      internal._TypeBuilderSetStructType(
        this.ref,
        index,
        types,
        packedTypes,
        mutables,
        fields.length,
      );
    });
  }
}

export class TypeBuilderError extends Error {
  constructor(
    public index: BinaryenIndex,
    public reason: TypeBuilderErrorReason,
  ) {
    super(`type builder error at index ${index} with reason ${reason}`);
  }
}

/** Construct a recursive type. */
export const buildType = (
  size: BinaryenIndex,
  f: (builder: TypeBuilder) => void,
): BinaryenHeapType[] => {
  return withAlloc(4 * (2 + size), (out) => {
    const errorIndex = out;
    const errorReason = out + 4;
    const heapTypes = out + 8;
    let success: Bool;
    const builder = internal._TypeBuilderCreate(size);
    try {
      f(new TypeBuilder(builder));
    } finally {
      success = internal._TypeBuilderBuildAndDispose(
        builder,
        heapTypes,
        errorIndex,
        errorReason,
      );
    }
    if (!success)
      throw new TypeBuilderError(
        internal.__i32_load(errorIndex),
        internal.__i32_load(errorReason),
      );
    const types: BinaryenHeapType[] = [];
    for (let i = 0; i < size; ++i)
      types.push(internal.__i32_load(heapTypes + 4 * i));
    return types;
  });
};
