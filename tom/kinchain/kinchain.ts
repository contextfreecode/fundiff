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
    setTimeout(
      () => {
        const angles = parseRow(field);
        const lengths = parseLengths();
        const chain = forward({ angles, lengths });
        render(chain);
      },
      0,
    );
  };
  update();
  field.addEventListener("click", (event) => update());
  field.addEventListener("keydown", (event) => update());
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
[ 1.6061517 -0.8207535 -0.8207535]
[ 1.6412461 -0.8583218 -0.8575862]
[ 1.676161   -0.89810216 -0.89588124]
[ 1.7109771 -0.9400804 -0.9356116]
[ 1.7457726 -0.9842271 -0.9767375]
[ 1.7806233 -1.0304965 -1.0192052]
[ 1.8156016 -1.0788243 -1.0629467]
[ 1.8507761 -1.1291264 -1.1078789]
[ 1.8862109 -1.1812975 -1.1539043]
[ 1.9219662 -1.2352092 -1.2009108]
[ 1.9580976 -1.2907088 -1.2487735]
[ 1.9946566 -1.3476175 -1.2973561]
[ 2.031691  -1.4057294 -1.3465141]
[ 2.0692463 -1.464809  -1.3960975]
[ 2.1073651 -1.5245887 -1.4459558]
[ 2.1460907 -1.5847657 -1.4959428]
[ 2.1854675 -1.6449958 -1.5459235]
[ 2.2255435 -1.7048852 -1.5957806]
[ 2.2663736 -1.7639772 -1.6454233]
[ 2.3080232 -1.8217312 -1.6947963]
`.trim();

main();
