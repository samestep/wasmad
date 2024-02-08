import binaryen from "binaryen";
import { describe } from "vitest";
import * as util from "../util.js";
import { goldenfile } from "./util.js";

describe("buildType", () => {
  goldenfile("struct.wat", () => {
    const mod = new binaryen.Module();
    try {
      const [struct] = util.buildType(1, (builder) => {
        builder.setStructType(0, [
          {
            type: binaryen.i32,
            packedType: util.packedTypeNotPacked,
            mutable: false,
          },
        ]);
      });
      const type = util.typeFromHeapType(struct, false);
      mod.addFunction("foo", type, type, [], mod.local.get(0, type));
      return mod.emitText();
    } finally {
      mod.dispose();
    }
  });
});
