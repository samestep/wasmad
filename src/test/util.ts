import fs from "fs/promises";
import path from "path";
import url from "url";
import { expect, test } from "vitest";

export const dir = path.dirname(url.fileURLToPath(import.meta.url));

export const rmrf = async (dirname: string): Promise<void> => {
  await fs.rm(path.join(dir, dirname), { recursive: true, force: true });
};

export const mkdirp = async (dirname: string): Promise<void> => {
  await fs.mkdir(path.join(dir, dirname), { recursive: true });
};

export const slurp = async (filename: string): Promise<string> =>
  await fs.readFile(path.join(dir, filename), "utf8");

export const spit = async (filename: string, data: string): Promise<void> =>
  await fs.writeFile(path.join(dir, filename), data, "utf8");

export const goldenfile = (
  filename: string,
  f: () => string | Promise<string>,
): void =>
  test(filename, async (): Promise<void> => {
    const actual = await f();
    if (process.env.GOLDENFILE) {
      await spit(filename, actual);
      return;
    }
    const message = "rerun with GOLDENFILE=1 to overwrite";
    let expected: string;
    try {
      expected = await slurp(filename);
    } catch (error) {
      throw Error(`${message}: ${error}`);
    }
    expect(actual, message).toBe(expected);
  });
