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
[ 1.5368664  -0.83338237 -0.80939025]
[ 1.5142332 -0.8766613 -0.8328429]
[ 1.50222   -0.9151539 -0.8556762]
[ 1.4990746  -0.94943964 -0.8779394 ]
[ 1.5027242  -0.98042816 -0.8997735 ]
[ 1.5113255  -1.0090346  -0.92134535]
[ 1.5234523 -1.0360206 -0.9428048]
[ 1.5380745  -1.0619627  -0.96426916]
[ 1.5544736  -1.0872748  -0.98582304]
[ 1.572159  -1.1122437 -1.0075239]
[ 1.5908012 -1.1370608 -1.0294083]
[ 1.610183  -1.1618488 -1.0514973]
[ 1.6301646 -1.1866803 -1.073801 ]
[ 1.6506592 -1.2115928 -1.0963216]
[ 1.6716156 -1.2365985 -1.1190559]
[ 1.693006  -1.2616919 -1.141997 ]
[ 1.7148181 -1.2868552 -1.1651359]
[ 1.7370495 -1.3120611 -1.1884617]
[ 1.7597034 -1.3372759 -1.2119627]
[ 1.7827877 -1.3624601 -1.2356266]
`.trim();

main();
