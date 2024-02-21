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

/**
 * Do not use if `undefined` is a value of `V`.
 * @returns either `m.get(k)` or `f()`; in latter case, inserts result into `m`
 */
export const cached = <K, V>(k: K, m: Map<K, V>, f: () => V): V => {
  let v = m.get(k);
  if (v !== undefined) return v;
  v = f();
  m.set(k, v);
  return v;
};

/** Finds names similar to desired ones, to avoid conflicts. */
export class Names {
  private set: Set<string>;

  constructor() {
    this.set = new Set();
  }

  /** Add `name` to the set, throwing if it is already present. */
  add(name: string): void {
    if (this.set.has(name)) throw Error(`Name conflict: ${name}`);
    this.set.add(name);
  }

  /** Make up a name similar to `name`, add it to the set, and return it. */
  make(name: string): string {
    // TODO: be smarter to avoid pathological cases, maybe e.g. like this?
    // https://cs.stackexchange.com/a/39700
    for (let i = 1; this.set.has(name); ++i) name = `${name}${i}`;
    this.set.add(name);
    return name;
  }
}

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

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L107 */
  _BinaryenTypeStructref(): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L108 */
  _BinaryenTypeArrayref(): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L113 */
  _BinaryenTypeNullref(): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L114 */
  _BinaryenTypeNullExternref(): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L115 */
  _BinaryenTypeNullFuncref(): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L136 */
  _BinaryenPackedTypeNotPacked(): BinaryenPackedType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L160 */
  _BinaryenHeapTypeIsSignature(heapType: BinaryenHeapType): Bool;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L161 */
  _BinaryenHeapTypeIsStruct(heapType: BinaryenHeapType): Bool;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L162 */
  _BinaryenHeapTypeIsArray(heapType: BinaryenHeapType): Bool;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L168-L169 */
  _BinaryenStructTypeGetNumFields(heapType: BinaryenHeapType): BinaryenIndex;

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

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L176-L177 */
  _BinaryenArrayTypeGetElementType(heapType: BinaryenHeapType): binaryen.Type;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L178-L179 */
  _BinaryenArrayTypeGetElementPackedType(
    heapType: BinaryenHeapType,
  ): BinaryenPackedType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L180 */
  _BinaryenArrayTypeIsElementMutable(heapType: BinaryenHeapType): Bool;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L186 */
  _BinaryenTypeGetHeapType(type: binaryen.Type): BinaryenHeapType;

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L187 */
  _BinaryenTypeIsNullable(type: binaryen.Type): Bool;

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

  /** https://github.com/WebAssembly/binaryen/blob/version_116/src/binaryen-c.h#L3553-L3558 */
  _TypeBuilderSetArrayType(
    builder: TypeBuilderRef,
    index: BinaryenIndex,
    elementType: binaryen.Type,
    elementPackedType: BinaryenPackedType,
    elementMutable: Int,
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

export const structref: binaryen.Type = internal._BinaryenTypeStructref();

export const arrayref: binaryen.Type = internal._BinaryenTypeArrayref();

export const nullref: binaryen.Type = internal._BinaryenTypeNullref();

export const nullexternref: binaryen.Type =
  internal._BinaryenTypeNullExternref();

export const nullfuncref: binaryen.Type = internal._BinaryenTypeNullFuncref();

export const packedTypeNotPacked: BinaryenPackedType =
  internal._BinaryenPackedTypeNotPacked();

export const heapTypeIsSignature = (heapType: BinaryenHeapType): boolean =>
  !!internal._BinaryenHeapTypeIsSignature(heapType);

export const heapTypeIsStruct = (heapType: BinaryenHeapType): boolean =>
  !!internal._BinaryenHeapTypeIsStruct(heapType);

export const heapTypeIsArray = (heapType: BinaryenHeapType): boolean =>
  !!internal._BinaryenHeapTypeIsArray(heapType);

export interface Field {
  type: binaryen.Type;
  packedType: BinaryenPackedType;
  mutable: boolean;
}

export const structTypeGetFields = (heapType: BinaryenHeapType): Field[] => {
  const n = internal._BinaryenStructTypeGetNumFields(heapType);
  const fields = [];
  for (let i = 0; i < n; ++i)
    fields.push({
      type: internal._BinaryenStructTypeGetFieldType(heapType, i),
      packedType: internal._BinaryenStructTypeGetFieldPackedType(heapType, i),
      mutable: !!internal._BinaryenStructTypeIsFieldMutable(heapType, i),
    });
  return fields;
};

export const arrayTypeGetElementType = (
  heapType: BinaryenHeapType,
): binaryen.Type => internal._BinaryenArrayTypeGetElementType(heapType);

export const arrayTypeGetElementPackedType = (
  heapType: BinaryenHeapType,
): BinaryenPackedType =>
  internal._BinaryenArrayTypeGetElementPackedType(heapType);

export const arrayTypeIsElementMutable = (
  heapType: BinaryenHeapType,
): boolean => !!internal._BinaryenArrayTypeIsElementMutable(heapType);

export const typeGetHeapType = (type: binaryen.Type): BinaryenHeapType =>
  internal._BinaryenTypeGetHeapType(type);

export const typeIsNullable = (type: binaryen.Type): boolean =>
  !!internal._BinaryenTypeIsNullable(type);

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

  setArrayType(
    index: BinaryenIndex,
    elementType: binaryen.Type,
    elementPackedType: BinaryenPackedType,
    elementMutable: boolean,
  ): void {
    internal._TypeBuilderSetArrayType(
      this.ref,
      index,
      elementType,
      elementPackedType,
      toBool(elementMutable),
    );
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

/** Construct heap types. */
export const buildTypes = (
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

export const buildStructType = (fields: Field[]): BinaryenHeapType => {
  const [heapType] = buildTypes(1, (builder) => {
    builder.setStructType(0, fields);
  });
  return heapType;
};

export const buildArrayType = (
  elementType: binaryen.Type,
  elementPackedType: BinaryenPackedType,
  elementMutable: boolean,
): BinaryenHeapType => {
  const [heapType] = buildTypes(1, (builder) => {
    builder.setArrayType(0, elementType, elementPackedType, elementMutable);
  });
  return heapType;
};
