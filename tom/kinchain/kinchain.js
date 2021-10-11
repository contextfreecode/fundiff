function forward({ angles , lengths  }) {
    let end = {
        angle: 0,
        point: {
            x: 0,
            y: 0
        }
    };
    return lengths.map((length, index)=>{
        const angle = angles[index] + end.angle;
        end = {
            angle,
            point: {
                x: end.point.x + length * Math.cos(angle),
                y: end.point.y + length * Math.sin(angle)
            }
        };
        return end;
    });
}
function main1() {
    const seq = parseSeq();
    console.log(seq);
    const chain = forward({
        angles: seq.angleFrames.slice(-1)[0],
        lengths: seq.lengths
    });
    console.log(chain);
    render(chain);
}
function parseSeq() {
    const field = document.getElementById("field");
    const lines = field.value.split("\n");
    let rows = lines.map((line)=>{
        const texts = line.replaceAll(/[^-+\d\.e]+/g, " ").trim().split(/\s+/);
        const vals = texts.map((text)=>parseFloat(text)
        );
        return vals;
    });
    rows = rows.filter((row)=>row.length == rows[0].length
    );
    return {
        angleFrames: rows.slice(1),
        lengths: rows[0]
    };
}
function render(chain) {
    const path = document.getElementById("path");
    const coords = chain.flatMap(({ point: { x , y  }  })=>[
            x,
            y
        ]
    );
    console.log(coords);
    const d = `M 0 0 L ${coords.join(" ")}`;
    console.log(d);
    path.setAttribute("d", d);
}
main1();
export { main1 as main };
