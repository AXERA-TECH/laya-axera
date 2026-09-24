// Run with: node tests/js/tank_sim.test.js
const assert = require("assert");
const { TankSim: S } = require("../../laya_axera/web/tank.js");
let pass = 0;
const t = (name, fn) => { fn(); pass++; console.log("ok -", name); };
const run = (g, ms, now, controls = {}) => { for (let i = 0; i < ms / 10; i++) { now += 10; S.update(g, 10, now, controls); } return now; };

t("layout is 13x13, spawns empty, mirror symmetric", () => {
  const g = S.create("easy", 0);
  assert.equal(g.grid.length, 13); assert.ok(g.grid.every((r) => r.length === 13));
  assert.equal(g.grid[12][6], S.EMPTY); assert.equal(g.grid[0][6], S.EMPTY);
  for (let y = 0; y < 13; y++) for (let x = 0; x < 13; x++) {
    assert.equal(g.grid[y][x], g.grid[12 - y][x], "top/bottom mirror");
    assert.equal(g.grid[y][x], g.grid[y][12 - x], "left/right mirror");
  }
});

t("player steps one tile per move and turns in place when blocked", () => {
  const g = S.create("easy", 0);
  let now = run(g, 20, 0, { dir: "north" });           // tap: one tile
  now = run(g, 300, now);
  assert.equal(g.tanks.you.y, 11);
  now = run(g, 20, now, { dir: "west" });
  now = run(g, 300, now);
  assert.equal(g.tanks.you.x, 5, "one tile west");
  now = run(g, 1000, now, { dir: "west" });            // held: keeps going until the brick at x=3
  assert.equal(g.tanks.you.x, 4);
  const g2 = S.create("easy", 0);
  g2.grid[11][6] = S.STEEL;
  run(g2, 300, 0, { dir: "north" });
  assert.equal(g2.tanks.you.y, 12, "blocked: stays"); assert.equal(g2.tanks.you.dir, "north", "but turns");
});

t("a shot breaks the first brick in line and stops", () => {
  const g = S.create("easy", 0);
  g.grid[9][6] = S.EMPTY; g.grid[3][6] = S.EMPTY;       // clear the central bunkers on column 6
  g.grid[8][6] = S.BRICK; g.grid[7][6] = S.BRICK;
  let now = 2000;                                        // past the spawn shield window for firing checks
  S.fire(g, g.tanks.you, now, 0, 10);
  now = run(g, 1500, now);
  assert.equal(g.grid[8][6], S.EMPTY, "first brick broken");
  assert.equal(g.grid[7][6], S.BRICK, "second brick untouched");
  assert.equal(g.bullets.length, 0);
});

t("steel stops shots without breaking", () => {
  const g = S.create("easy", 0);
  g.grid[9][6] = S.STEEL;
  S.fire(g, g.tanks.you, 2000, 0, 10);
  run(g, 1000, 2000);
  assert.equal(g.grid[9][6], S.STEEL);
});

t("a hit costs a life, then respawn with a shield; shielded tanks shrug off hits", () => {
  const g = S.create("easy", 0);
  for (let y = 1; y < 12; y++) g.grid[y][6] = S.EMPTY;
  let now = 5000;                                         // both shields long expired
  g.tanks.you.shieldUntil = g.tanks.ai.shieldUntil = 0;
  S.fire(g, g.tanks.you, now, 0, 12);
  now = run(g, 1500, now);
  assert.equal(g.tanks.ai.lives, 2); assert.ok(g.events.some((e) => e.type === "boom"));
  now = run(g, 1200, now);
  assert.equal(g.tanks.ai.dead, false, "respawned");
  assert.ok(g.tanks.ai.shieldUntil > now, "with a shield");
  S.fire(g, g.tanks.you, now, 0, 12);
  run(g, 900, now);
  assert.equal(g.tanks.ai.lives, 2, "shield absorbed the second shot");
});

t("head-on shots cancel even at high speed", () => {
  const g = S.create("easy", 0);
  for (let y = 1; y < 12; y++) g.grid[y][6] = S.EMPTY;
  g.tanks.you.shieldUntil = g.tanks.ai.shieldUntil = 0;
  S.fire(g, g.tanks.you, 3000, 0, 12);
  S.fire(g, g.tanks.ai, 3000, 0, 12);
  run(g, 1500, 3000);
  assert.equal(g.tanks.you.lives, 3); assert.equal(g.tanks.ai.lives, 3);
  assert.equal(g.bullets.length, 0);
});

t("one bullet in flight per tank, and cooldown is honoured", () => {
  const g = S.create("easy", 0);
  assert.equal(S.fire(g, g.tanks.you, 100, 500, 1), true);
  assert.equal(S.fire(g, g.tanks.you, 150, 500, 1), false, "bullet still flying");
  g.bullets.length = 0;
  assert.equal(S.fire(g, g.tanks.you, 300, 500, 1), false, "cooldown");
  assert.equal(S.fire(g, g.tanks.you, 700, 500, 1), true);
});

t("three hits end the match", () => {
  const g = S.create("easy", 0);
  for (let y = 1; y < 12; y++) g.grid[y][6] = S.EMPTY;
  let now = 5000;
  for (let i = 0; i < 3; i++) {
    while (g.tanks.ai.dead) now = run(g, 50, now);
    g.tanks.ai.shieldUntil = 0;
    S.fire(g, g.tanks.you, now, 0, 12);
    now = run(g, 1400, now);
  }
  assert.equal(g.over, "you");
});

t("AI decisions: shoot turns and fires, move turns and steps; rate limit gate", () => {
  const g = S.create("hell", 0);
  assert.equal(S.aiReady(g, 100), false, "opening delay");
  assert.equal(S.aiReady(g, 800), true);
  S.applyAi(g, { action: "east", mode: "move" }, 1000);
  assert.equal(g.tanks.ai.dir, "east"); assert.ok(g.tanks.ai.move);
  run(g, 200, 1000);
  assert.equal(g.tanks.ai.x, 7);
  S.applyAi(g, { action: "south", mode: "shoot" }, 1300);
  assert.equal(g.tanks.ai.dir, "south");
  assert.equal(g.bullets.filter((b) => b.owner === "ai").length, 1);
  const p = S.aiPayload(g);
  assert.deepEqual(Object.keys(p).sort(), ["ai", "bullets", "grid", "player"]);
  assert.equal(p.bullets[0].owner, "ai");
});

t("tanks never share a tile", () => {
  const g = S.create("easy", 0);
  g.tanks.ai.x = 6; g.tanks.ai.y = 11; g.tanks.ai.px = 6; g.tanks.ai.py = 11;
  run(g, 300, 0, { dir: "north" });
  assert.equal(g.tanks.you.y, 12);
});
console.log(`\n${pass} passed`);
