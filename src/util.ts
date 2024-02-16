import binaryen from "binaryen";

/**
 * Throws if `x === undefined`.
 * @returns `x`
 * @param f called if `x === undefined`, to produce error message
 */
export const unwrap = <T>(x: T | undefined, f?: () => string): T => {
  if (x === undefined)
    throw Error((f ?? (() => "called `unwrap` with `undefined`"))());
  return x;
};

export const funcIndicesByName = (
  mod: binaryen.Module,
): Map<string, number> => {
  const n = mod.getNumFunctions();
  const indices = new Map<string, number>();
  for (let i = 0; i < n; ++i)
    indices.set(binaryen.getFunctionInfo(mod.getFunctionByIndex(i)).name, i);
  return indices;
};

type Bool = number;

type Int = number;

type Pointer<T> = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L85-L89 */
export type BinaryenIndex = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L132-L134 */
export type BinaryenPackedType = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L140-L142 */
export type BinaryenHeapType = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L251 */
type BinaryenModuleRef = number;

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3507-L3514 */
type TypeBuilderRef = number;

interface Internal {
  _malloc(size: number): Pointer<any>;
  __i32_store8(pointer: Pointer<number>, value: number): void;
  __i32_store(pointer: Pointer<number>, value: number): void;
  __i32_load(pointer: Pointer<number>): number;
  _free(pointer: Pointer<any>): void;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L136 */
  _BinaryenPackedTypeNotPacked(): BinaryenPackedType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L170-L171 */
  _BinaryenStructTypeGetFieldType(
    heapType: BinaryenHeapType,
    index: BinaryenIndex,
  ): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L172-L173 */
  _BinaryenStructTypeGetFieldPackedType(
    heapType: BinaryenHeapType,
    index: BinaryenIndex,
  ): BinaryenPackedType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L174-L175 */
  _BinaryenStructTypeIsFieldMutable(
    heapType: BinaryenHeapType,
    index: BinaryenIndex,
  ): Bool;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L188-L189 */
  _BinaryenTypeFromHeapType(
    heapType: BinaryenHeapType,
    nullable: Bool,
  ): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L1054-L1058 */
  _BinaryenStructNew(
    module: BinaryenModuleRef,
    operands: Pointer<binaryen.ExpressionRef>,
    numOperands: BinaryenIndex,
    type: BinaryenHeapType,
  ): binaryen.ExpressionRef;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L1059-L1063 */
  _BinaryenStructGet(
    module: BinaryenModuleRef,
    index: BinaryenIndex,
    ref: binaryen.ExpressionRef,
    type: binaryen.Type,
    signed_: Bool,
  ): binaryen.ExpressionRef;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3518-L3519 */
  _TypeBuilderErrorReasonSelfSupertype(): TypeBuilderErrorReason;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3520-L3522 */
  _TypeBuilderErrorReasonInvalidSupertype(): TypeBuilderErrorReason;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3523-L3525 */
  _TypeBuilderErrorReasonForwardSupertypeReference(): TypeBuilderErrorReason;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3526-L3528 */
  _TypeBuilderErrorReasonForwardChildReference(): TypeBuilderErrorReason;

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

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3559-L3562 */
  _TypeBuilderGetTempHeapType(
    builder: TypeBuilderRef,
    index: BinaryenIndex,
  ): BinaryenHeapType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3567-L3570 */
  _TypeBuilderGetTempRefType(
    builder: TypeBuilderRef,
    heapType: BinaryenHeapType,
    nullable: Int,
  ): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3578-L3582 */
  _TypeBuilderCreateRecGroup(
    builder: TypeBuilderRef,
    index: BinaryenIndex,
    length: BinaryenIndex,
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

export interface Field {
  type: binaryen.Type;
  packedType: BinaryenPackedType;
  mutable: boolean;
}

export const structTypeGetField = (
  heapType: BinaryenHeapType,
  index: BinaryenIndex,
): Field => ({
  type: internal._BinaryenStructTypeGetFieldType(heapType, index),
  packedType: internal._BinaryenStructTypeGetFieldPackedType(heapType, index),
  mutable: !!internal._BinaryenStructTypeIsFieldMutable(heapType, index),
});

export const typeFromHeapType = (
  heapType: BinaryenHeapType,
  nullable: boolean,
): binaryen.Type =>
  internal._BinaryenTypeFromHeapType(heapType, toBool(nullable));

export const structNew = (
  mod: binaryen.Module,
  operands: binaryen.ExpressionRef[],
  type: BinaryenHeapType,
): binaryen.ExpressionRef =>
  withAlloc(4 * operands.length, (pointer) => {
    for (let i = 0; i < operands.length; ++i)
      internal.__i32_store(pointer + 4 * i, operands[i]);
    return internal._BinaryenStructNew(mod.ptr, pointer, operands.length, type);
  });

export const structGet = (
  mod: binaryen.Module,
  index: BinaryenIndex,
  ref: binaryen.ExpressionRef,
  type: binaryen.Type,
  signed?: boolean,
): binaryen.ExpressionRef =>
  internal._BinaryenStructGet(
    mod.ptr,
    index,
    ref,
    type,
    toBool(signed ?? false),
  );

/** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3516 */
export enum TypeBuilderErrorReason {
  SelfSupertype = internal._TypeBuilderErrorReasonSelfSupertype(),
  InvalidSupertype = internal._TypeBuilderErrorReasonInvalidSupertype(),
  ForwardSupertypeReference = internal._TypeBuilderErrorReasonForwardSupertypeReference(),
  ForwardChildReference = internal._TypeBuilderErrorReasonForwardChildReference(),
}

export class TypeBuilderError extends Error {
  constructor(
    public index: BinaryenIndex,
    public reason: TypeBuilderErrorReason,
  ) {
    super(
      `type builder error at index ${index} with reason ${TypeBuilderErrorReason[reason]}`,
    );
  }
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

  getTempHeapType(index: BinaryenIndex): BinaryenHeapType {
    return internal._TypeBuilderGetTempHeapType(this.ref, index);
  }

  getTempRefType(heapType: BinaryenHeapType, nullable: boolean): binaryen.Type {
    return internal._TypeBuilderGetTempRefType(
      this.ref,
      heapType,
      toBool(nullable),
    );
  }

  createRecGroup(index: BinaryenIndex, length: BinaryenIndex): void {
    internal._TypeBuilderCreateRecGroup(this.ref, index, length);
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
