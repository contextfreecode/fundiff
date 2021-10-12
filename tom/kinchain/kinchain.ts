type Arm = { angles: number[]; lengths: number[] };

type Transform = { angle: number; point: Point };

type Point = { x: number; y: number };

function forward({ angles, lengths }: Arm): Transform[] {
  let end = { angle: 0, point: { x: 0, y: 0 } } as Transform;
  return lengths.map(
    (length, index) => {
      const angle = angles[index] + end.angle;
      end = {
        angle,
        point: {
          x: end.point.x + length * Math.cos(angle),
          y: end.point.y + length * Math.sin(angle),
        },
      };
      return end;
    },
  );
}

export function main() {
  const field = document.getElementById("field") as HTMLTextAreaElement;
  if (!field.value.trim()) {
    field.value = defaultText;
  }
  const update = () => {
    const angles = parseRow(field);
    const lengths = parseLengths();
    const chain = forward({ angles, lengths });
    render(chain);
  };
  update();
  document.addEventListener("selectionchange", (event) => {
    console.log(event);
    if (document.activeElement == field) {
      update();
    }
  });
}

function parseLine(line: string): number[] {
  const texts = line.replaceAll(/[^-+\d\.e]+/g, " ").trim().split(/\s+/);
  const vals = texts.map((text) => parseFloat(text));
  return vals;
}

function parseRow(field: HTMLTextAreaElement): number[] {
  const text: string = field.value;
  if (!text.trim()) {
    return [];
  }
  const index: number = field.selectionStart;
  return parseRowAt(text, index);
}

function parseRowAt(text: string, index: number): number[] {
  const begin = Math.max(0, text.lastIndexOf("\n", index - 1));
  let end = text.indexOf("\n", index);
  if (end < 0) {
    end = text.length;
  }
  if (begin <= 0) {
    return parseRowAt(text, end + 1);
  }
  const line = text.slice(begin, end);
  if (!line.trim()) {
    return parseRowAt(text, begin - 1);
  }
  return parseLine(line);
}

function parseLengths(): number[] {
  const field = document.getElementById("field") as HTMLTextAreaElement;
  const lines = field.value.split("\n") as string[];
  return parseLine(lines[0]);
}

function render(chain: Transform[]) {
  const path = document.getElementById("path")!;
  const coords = chain.flatMap(({ point: { x, y } }) => [x, y]);
  const d = chain.length ? `M 0 0 L ${coords.join(" ")}` : "";
  path.setAttribute("d", d);
}

const defaultText = `
[1.  1.  0.5]
[ 1.5707964 -0.7853982 -0.7853982]
[ 1.657082  -0.7853982 -0.8106707]
[ 1.7370871  -0.7902707  -0.83817226]
[ 1.8103027  -0.80053526 -0.8680589 ]
[ 1.8767295  -0.81637144 -0.90036815]
[ 1.9368242  -0.83765423 -0.9350354 ]
[ 1.9913634 -0.8640406 -0.9719254]
[ 2.0412838  -0.89506704 -1.0108644 ]
[ 2.0875497 -0.9302287 -1.0516642]
[ 2.1310637 -0.9690303 -1.0941368]
[ 2.1726222 -1.0110115 -1.1381   ]
[ 2.2128997 -1.0557544 -1.1833789]
[ 2.252452  -1.1028818 -1.2298048]
[ 2.2917278 -1.1520497 -1.2772129]
[ 2.331085  -1.2029384 -1.3254415]
[ 2.3708065 -1.2552439 -1.3743306]
[ 2.4111166 -1.3086678 -1.4237224]
[ 2.4521964 -1.362908  -1.4734622]
[ 2.4942    -1.4176477 -1.5234015]
[ 2.537274  -1.4725388 -1.5734015]
[ 2.5815868 -1.5271732 -1.62334  ]
`.trim();

main();
