type Arm = { angles: number[]; lengths: number[] };

type Transform = { angle: number; point: Point };

type Point = { x: number; y: number };

type Seq = { angleFrames: number[][]; lengths: number[] };

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
  const lengths = parseLengths();
  const field = document.getElementById("field") as HTMLTextAreaElement;
  const update = () => {
    const angles = parseRow(field);
    const chain = forward({ angles, lengths });
    render(chain);
  };
  update();
  document.addEventListener("selectionchange", (event) => {
    if (event.target == field) {
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
  const d = `M 0 0 L ${coords.join(" ")}`;
  path.setAttribute("d", d);
}

main();
