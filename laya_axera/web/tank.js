"use strict";
// Tank duel. The arena runs here in the page; only the AI tank's decisions go to
// the server, where Laya picks a direction on the NPU. Difficulty limits how
// often the AI may decide and how fast it moves and shoots, so a person can win.

const TankSim = (() => {
  const N = 13;
  const EMPTY = 0, BRICK = 1, STEEL = 2;
  const DIRS = { north: [0, -1], south: [0, 1], west: [-1, 0], east: [1, 0] };
  const LEVELS = {
    easy: { label: "简单", think: 900, move: 300, shotSpeed: 6, cooldown: 900 },
    hard: { label: "困难", think: 450, move: 210, shotSpeed: 9, cooldown: 500 },
    hell: { label: "地狱", think: 220, move: 150, shotSpeed: 12, cooldown: 280 },
  };
  const PLAYER = { move: 170, shotSpeed: 11, cooldown: 330 };
  const LIVES = 3;
  const RESPAWN_MS = 1100;
  const SHIELD_MS = 1600;
  const HIT_R = 0.42;
  // An original, mirror-symmetric arena so neither spawn is favoured.
  const LAYOUT = [
    ".............",
    "..##.....##..",
    "..#..S.S..#..",
    ".....###.....",
    ".SS.......SS.",
    "...#.#.#.#...",
    ".#...S.S...#.",
    "...#.#.#.#...",
    ".SS.......SS.",
    ".....###.....",
    "..#..S.S..#..",
    "..##.....##..",
    ".............",
  ];
  const SPAWN = { you: { x: 6, y: 12, dir: "north" }, ai: { x: 6, y: 0, dir: "south" } };

  function makeTank(side, now) {
    const s = SPAWN[side];
    return {
      side, x: s.x, y: s.y, px: s.x, py: s.y, dir: s.dir, move: null, lives: LIVES,
      cooldownUntil: 0, dead: false, respawnAt: 0, shieldUntil: now + SHIELD_MS, tread: 0,
    };
  }

  function create(level = "easy", now = 0) {
    return {
      level,
      grid: LAYOUT.map((row) => [...row].map((c) => (c === "#" ? BRICK : c === "S" ? STEEL : EMPTY))),
      tanks: { you: makeTank("you", now), ai: makeTank("ai", now) },
      bullets: [],
      events: [],
      over: null,
      aiBusy: false,
      aiNextAt: now + 700,
    };
  }

  function tileFree(g, x, y, self) {
    if (x < 0 || y < 0 || x >= N || y >= N || g.grid[y][x] !== EMPTY) return false;
    for (const t of Object.values(g.tanks)) {
      if (t === self || t.dead) continue;
      if (t.x === x && t.y === y) return false;
      if (t.move && t.move.tx === x && t.move.ty === y) return false;
    }
    return true;
  }

  // Face `dir`; start a one-tile move if the tile ahead is free.
  function steer(g, tank, dir, now, moveMs) {
    if (tank.dead || tank.move || !DIRS[dir]) return false;
    tank.dir = dir;
    const [dx, dy] = DIRS[dir];
    const tx = tank.x + dx, ty = tank.y + dy;
    if (!tileFree(g, tx, ty, tank)) return false;
    tank.move = { fx: tank.x, fy: tank.y, tx, ty, t0: now, dur: moveMs };
    return true;
  }

  function fire(g, tank, now, cooldownMs, speed) {
    if (tank.dead || now < tank.cooldownUntil) return false;
    if (g.bullets.some((b) => b.owner === tank.side)) return false;
    const [dx, dy] = DIRS[tank.dir];
    g.bullets.push({ x: tank.px + dx * 0.45, y: tank.py + dy * 0.45, dx, dy, speed, owner: tank.side });
    tank.cooldownUntil = now + cooldownMs;
    g.events.push({ type: "shot", x: tank.px + dx * 0.55, y: tank.py + dy * 0.55 });
    return true;
  }

  function hit(g, tank, now) {
    tank.lives -= 1;
    tank.dead = true;
    tank.move = null;
    tank.respawnAt = now + RESPAWN_MS;
    g.events.push({ type: "boom", side: tank.side, x: tank.px, y: tank.py });
    if (tank.lives <= 0) g.over = tank.side === "you" ? "ai" : "you";
  }

  function moveTanks(g, dt, now) {
    for (const t of Object.values(g.tanks)) {
      if (t.dead) {
        const s = SPAWN[t.side];
        if (t.lives > 0 && now >= t.respawnAt && tileFree(g, s.x, s.y, t)) {
          Object.assign(t, { x: s.x, y: s.y, px: s.x, py: s.y, dir: s.dir, dead: false, shieldUntil: now + SHIELD_MS });
          g.events.push({ type: "spawn", side: t.side, x: s.x, y: s.y });
        }
        continue;
      }
      if (!t.move) continue;
      const k = Math.min(1, (now - t.move.t0) / t.move.dur);
      t.px = t.move.fx + (t.move.tx - t.move.fx) * k;
      t.py = t.move.fy + (t.move.ty - t.move.fy) * k;
      t.tread += dt / t.move.dur;
      if (k >= 1) {
        t.x = t.move.tx;
        t.y = t.move.ty;
        t.move = null;
      }
    }
  }

  function moveBullets(g, dt, now) {
    const before = new Map(g.bullets.map((b) => [b, [b.x, b.y]]));
    const kept = [];
    for (const b of g.bullets) {
      let left = (b.speed * dt) / 1000;
      let alive = true;
      while (alive && left > 0) {
        const step = Math.min(0.2, left);
        left -= step;
        b.x += b.dx * step;
        b.y += b.dy * step;
        const tx = Math.round(b.x), ty = Math.round(b.y);
        if (tx < 0 || ty < 0 || tx >= N || ty >= N) {
          alive = false;
          break;
        }
        const cell = g.grid[ty][tx];
        if (cell !== EMPTY) {
          if (cell === BRICK) {
            g.grid[ty][tx] = EMPTY;
            g.events.push({ type: "brick", x: tx, y: ty });
          } else {
            g.events.push({ type: "spark", x: b.x, y: b.y });
          }
          alive = false;
          break;
        }
        for (const t of Object.values(g.tanks)) {
          if (t.dead || t.side === b.owner) continue;
          if (Math.abs(b.x - t.px) < HIT_R && Math.abs(b.y - t.py) < HIT_R) {
            alive = false;
            if (now >= t.shieldUntil) hit(g, t, now);
            else g.events.push({ type: "spark", x: b.x, y: b.y });
            break;
          }
        }
      }
      if (alive) kept.push(b);
    }
    // Opposing shots cancel. Checked as a swept crossing: fast bullets on the same
    // line can pass each other within one frame.
    for (let i = 0; i < kept.length; i++) {
      for (let j = i + 1; j < kept.length; j++) {
        const a = kept[i], c = kept[j];
        if (a.gone || c.gone || a.owner === c.owner) continue;
        const [ax0, ay0] = before.get(a), [cx0, cy0] = before.get(c);
        const near = Math.hypot(a.x - c.x, a.y - c.y) < 0.35;
        const crossedX = Math.abs(a.y - c.y) < 0.3 && Math.sign(ax0 - cx0) !== Math.sign(a.x - c.x);
        const crossedY = Math.abs(a.x - c.x) < 0.3 && Math.sign(ay0 - cy0) !== Math.sign(a.y - c.y);
        if (near || crossedX || crossedY) {
          a.gone = c.gone = true;
          g.events.push({ type: "spark", x: (a.x + c.x) / 2, y: (a.y + c.y) / 2 });
        }
      }
    }
    g.bullets = kept.filter((b) => !b.gone);
  }

  // controls: { dir: compass name or null, fire: bool } for the human tank.
  function update(g, dt, now, controls) {
    if (g.over) return;
    moveTanks(g, dt, now);
    const you = g.tanks.you;
    if (controls && controls.dir) steer(g, you, controls.dir, now, PLAYER.move);
    if (controls && controls.fire) fire(g, you, now, PLAYER.cooldown, PLAYER.shotSpeed);
    moveBullets(g, dt, now);
  }

  function aiReady(g, now) {
    const ai = g.tanks.ai;
    return !g.over && !ai.dead && !ai.move && !g.aiBusy && now >= g.aiNextAt;
  }

  function aiPayload(g) {
    const you = g.tanks.you, ai = g.tanks.ai;
    return {
      grid: g.grid,
      ai: { x: ai.x, y: ai.y, dir: ai.dir, can_fire: !g.bullets.some((b) => b.owner === "ai") },
      player: { x: Math.round(you.px), y: Math.round(you.py), dir: you.dir, alive: !you.dead },
      bullets: g.bullets.map((b) => ({ x: b.x, y: b.y, dx: b.dx, dy: b.dy, owner: b.owner === "you" ? "player" : "ai" })),
    };
  }

  // Apply a decision from /api/tank/decide: "shoot" turns and fires, "move" turns and steps.
  function applyAi(g, decision, now) {
    const ai = g.tanks.ai;
    if (g.over || ai.dead || !DIRS[decision.action]) return;
    const lv = LEVELS[g.level];
    if (decision.mode === "shoot" && !ai.move) {
      ai.dir = decision.action;
      fire(g, ai, now, lv.cooldown, lv.shotSpeed);
    } else {
      steer(g, ai, decision.action, now, lv.move);
    }
  }

  return { N, EMPTY, BRICK, STEEL, DIRS, LEVELS, PLAYER, LIVES, create, update, steer, fire, aiReady, aiPayload, applyAi };
})();

const TankView = (() => {
  const ANGLE = { north: 0, east: Math.PI / 2, south: Math.PI, west: -Math.PI / 2 };
  const S = TankSim;

  function drawBrick(ctx, x, y, T) {
    const color = effectiveTheme() === "dark" ? "#a8533a" : "#b9603f";
    const d = Math.round(T * 0.12);
    GFX.block(ctx, x + 2, y + 2, T - 4, T - 4, color, { depth: d, radius: 4 });
    const top = T - 4 - d;
    ctx.strokeStyle = "rgba(60,20,10,0.45)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 1; i < 3; i++) {
      const yy = y + 2 + (top * i) / 3;
      ctx.moveTo(x + 4, yy);
      ctx.lineTo(x + T - 4, yy);
    }
    for (let i = 0; i < 3; i++) {
      const y0 = y + 2 + (top * i) / 3, y1 = y + 2 + (top * (i + 1)) / 3;
      const xx = x + (i % 2 ? T * 0.33 : T * 0.62);
      ctx.moveTo(xx, y0);
      ctx.lineTo(xx, y1);
    }
    ctx.stroke();
  }

  function drawSteel(ctx, x, y, T) {
    const color = effectiveTheme() === "dark" ? "#7d8898" : "#98a3b3";
    const d = Math.round(T * 0.12);
    GFX.block(ctx, x + 2, y + 2, T - 4, T - 4, color, { depth: d, radius: 3 });
    for (const [rx, ry] of [[0.22, 0.2], [0.78, 0.2], [0.22, 0.62], [0.78, 0.62]]) {
      GFX.sphere(ctx, x + T * rx, y + T * ry, T * 0.05, GFX.shade(color, 0.1));
    }
  }

  function drawTank(ctx, t, color, T, now) {
    const cx = (t.px + 0.5) * T, cy = (t.py + 0.5) * T, s = T * 0.84;
    GFX.floorShadow(ctx, cx + 4, cy + 7, s * 0.62, s * 0.36, 0.42);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(ANGLE[t.dir]);
    const tw = s * 0.24, th = s * 0.94;
    for (const side of [-1, 1]) {
      const tx = side * (s / 2 - tw / 2);
      ctx.fillStyle = "#1f2227";
      GFX.rr(ctx, tx - tw / 2, -th / 2, tw, th, 4);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.lineWidth = 2;
      const gap = th / 7, off = ((t.tread % 1) * gap + gap) % gap;
      ctx.beginPath();
      for (let yy = -th / 2 + off; yy < th / 2; yy += gap) {
        ctx.moveTo(tx - tw / 2 + 2, yy);
        ctx.lineTo(tx + tw / 2 - 2, yy);
      }
      ctx.stroke();
    }
    GFX.block(ctx, -s * 0.29, -s * 0.36, s * 0.58, s * 0.72, color, { depth: 5, radius: 6, shadow: false });
    const barrel = ctx.createLinearGradient(-s * 0.07, 0, s * 0.07, 0);
    barrel.addColorStop(0, GFX.shade(color, -0.5));
    barrel.addColorStop(0.4, GFX.shade(color, 0.25));
    barrel.addColorStop(1, GFX.shade(color, -0.45));
    ctx.fillStyle = barrel;
    ctx.fillRect(-s * 0.07, -s * 0.6, s * 0.14, s * 0.46);
    GFX.sphere(ctx, 0, 0, s * 0.21, GFX.shade(color, 0.06));
    ctx.restore();
    if (now < t.shieldUntil) {
      const a = 0.35 + 0.25 * Math.sin(now / 90);
      ctx.strokeStyle = GFX.alpha("#9fd8ff", a);
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, s * 0.62, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // game: a TankSim state or null for the idle arena; view: { simNow, level, flashes, fx }
  function render(canvas, game, view) {
    const { simNow = 0, level = "easy", flashes = [], fx = null } = view || {};
    const ctx = canvas.getContext("2d");
    const W = canvas.width, T = W / S.N;
    const dark = effectiveTheme() === "dark";
    const now = performance.now();
    GFX.floor(ctx, W, W, dark ? "#1b1e23" : "#d8d1bf");
    ctx.strokeStyle = dark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < S.N; i++) {
      ctx.moveTo(i * T + 0.5, 0); ctx.lineTo(i * T + 0.5, W);
      ctx.moveTo(0, i * T + 0.5); ctx.lineTo(W, i * T + 0.5);
    }
    ctx.stroke();
    const g = game || S.create(level, 0);
    for (let y = 0; y < S.N; y++) {
      for (let x = 0; x < S.N; x++) {
        if (g.grid[y][x] === S.BRICK) drawBrick(ctx, x * T, y * T, T);
        else if (g.grid[y][x] === S.STEEL) drawSteel(ctx, x * T, y * T, T);
      }
    }
    if (game) {
      const you = game.tanks.you, ai = game.tanks.ai;
      if (!you.dead) drawTank(ctx, you, GFX.token("--blue"), T, simNow);
      if (!ai.dead) drawTank(ctx, ai, GFX.token("--orange"), T, simNow);
      for (const b of game.bullets) {
        const bx = (b.x + 0.5) * T, by = (b.y + 0.5) * T;
        GFX.glow(ctx, bx, by, T * 0.34, b.owner === "you" ? "#8fd0ff" : "#ffcf6e", 0.6);
        GFX.sphere(ctx, bx, by, T * 0.09, b.owner === "you" ? "#cfeaff" : "#ffe7a8");
      }
    } else {
      drawTank(ctx, { ...S.create(level, 0).tanks.you, shieldUntil: 0 }, GFX.token("--blue"), T, 0);
      drawTank(ctx, { ...S.create(level, 0).tanks.ai, shieldUntil: 0 }, GFX.token("--orange"), T, 0);
    }
    for (let i = flashes.length - 1; i >= 0; i--) {
      const f = flashes[i], k = (now - f.t0) / f.dur;
      if (k >= 1) { flashes.splice(i, 1); continue; }
      GFX.glow(ctx, f.x, f.y, f.r * (0.4 + k * 0.6), f.color, 0.85 * (1 - k));
    }
    if (fx) fx.draw(ctx, now);
  }

  return { render };
})();

if (typeof module !== "undefined") module.exports = { TankSim, TankView };

if (typeof document !== "undefined") {
  (() => {
    const S = TankSim;
    const KEY_DIR = { up: "north", down: "south", left: "west", right: "east" };
    const DIR_TEXT = { north: "↑ 上", south: "↓ 下", west: "← 左", east: "→ 右" };
    const REASON = { aim: "瞄准开火", chase: "追击", breach: "打掉砖墙", dodge: "躲子弹", counter: "对射拦截", wander: "巡逻" };
    const fx = new GFX.Particles();
    let game = null, running = false, paused = false, gen = 0;
    let level = "easy", simNow = 0, lastFrame = 0, raf = 0;
    let decisions = 0, lastDecisionAt = 0;
    const flashes = [];
    const held = [];
    let fireQueued = false;

    function heartText(n) {
      return "♥".repeat(Math.max(0, n)) + "♡".repeat(Math.max(0, S.LIVES - n));
    }

    function initRows() {
      const box = $("k-rows");
      box.innerHTML = "";
      for (const d of ["north", "south", "west", "east"]) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.id = "kdir-" + d;
        row.innerHTML = `<span class="opt">${DIR_TEXT[d]}</span>
          <span class="track"><span class="fill"></span></span>
          <span class="val">–</span>`;
        box.appendChild(row);
      }
    }

    function showDecision(d) {
      for (const dir of ["north", "south", "west", "east"]) {
        const row = $("kdir-" + dir);
        const p = d.probabilities[dir] || 0;
        const chosen = d.action === dir;
        row.classList.toggle("chosen", chosen);
        row.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
        row.querySelector(".val").textContent = pct(p);
        const shoot = d.modes && d.modes[dir] === "shoot" ? " · 开火" : "";
        row.querySelector(".opt").textContent = (chosen ? "✓ " : "") + DIR_TEXT[dir] + shoot;
      }
      pressFx("k-" + d.action);
      if (d.mode === "shoot") pressFx("k-fire");
      $("k-note").textContent = `AI：${REASON[d.reason] || d.reason}`;
      decisions += 1;
      const now = performance.now();
      $("k-s-dec").textContent = decisions;
      $("k-s-inf").innerHTML = d.inference_ms.toFixed(0) + "<small> ms</small>";
      if (lastDecisionAt) $("k-s-gap").innerHTML = (now - lastDecisionAt).toFixed(0) + "<small> ms</small>";
      lastDecisionAt = now;
    }

    function updateHud() {
      $("k-you-lives").textContent = heartText(game ? game.tanks.you.lives : S.LIVES);
      $("k-ai-lives").textContent = heartText(game ? game.tanks.ai.lives : S.LIVES);
    }

    async function askAi(myGen) {
      game.aiBusy = true;
      try {
        const d = await api("/api/tank/decide", { model: state.current, ...S.aiPayload(game) });
        if (myGen !== gen || !running) return;
        S.applyAi(game, d, simNow);
        showDecision(d);
      } catch (err) {
        if (myGen === gen) {
          showError("tank-error", err.message);
          setPaused(true);
        }
      } finally {
        if (myGen === gen && game) {
          game.aiBusy = false;
          game.aiNextAt = simNow + S.LEVELS[game.level].think;
        }
      }
    }

    function handleEvents(T) {
      const colors = { you: GFX.token("--blue"), ai: GFX.token("--orange") };
      for (const e of game.events) {
        const cx = (e.x + 0.5) * T, cy = (e.y + 0.5) * T;
        if (e.type === "boom") {
          flashes.push({ x: cx, y: cy, t0: performance.now(), dur: 520, r: T * 1.1, color: "#ffb14a" });
          fx.burst(cx, cy, colors[e.side], 26, 340, T * 0.14);
          fx.burst(cx, cy, "#ffd257", 16, 260, T * 0.1);
        } else if (e.type === "brick") {
          fx.burst(cx, cy, "#b45a3c", 14, 220, T * 0.12);
        } else if (e.type === "spark") {
          fx.burst(cx, cy, "#ffe08a", 8, 180, T * 0.07);
        } else if (e.type === "shot") {
          flashes.push({ x: cx, y: cy, t0: performance.now(), dur: 110, r: T * 0.32, color: "#fff2b0" });
        }
      }
      game.events.length = 0;
      updateHud();
    }

    function heldDir() {
      return held.length ? KEY_DIR[held[held.length - 1]] : null;
    }

    function frame(t) {
      raf = 0;
      if ($("view-tank").hidden) {
        if (running && !paused) setPaused(true);
        lastFrame = 0;
        return;
      }
      const dt = lastFrame ? Math.min(50, t - lastFrame) : 16;
      lastFrame = t;
      if (running && !paused) {
        simNow += dt;
        S.update(game, dt, simNow, { dir: heldDir(), fire: fireQueued });
        fireQueued = false;
        if (S.aiReady(game, simNow)) askAi(gen);
        handleEvents($("kboard").width / S.N);
        if (game.over) finish();
      }
      TankView.render($("kboard"), game, { simNow, level, flashes, fx });
      if ((running && !paused) || fx.alive || flashes.length) raf = requestAnimationFrame(frame);
    }

    function kick() {
      if (!raf) raf = requestAnimationFrame(frame);
    }

    function finish() {
      running = false;
      $("tank-toggle").disabled = true;
      const won = game.over === "you";
      $("k-overlay-title").textContent = won ? "你赢了" : "AI 赢了";
      $("k-overlay-sub").textContent =
        `难度：${S.LEVELS[game.level].label}　剩余生命 你 ${game.tanks.you.lives} : ${game.tanks.ai.lives} AI。点「开始对战」再来一局。`;
      $("k-overlay").hidden = false;
      $("tank-new").textContent = "开始对战";
    }

    function setPaused(p) {
      paused = p;
      $("tank-toggle").textContent = p ? "继续" : "暂停";
      if (!p) kick();
    }

    function start() {
      hide("tank-error");
      gen += 1;
      simNow = 0;
      game = S.create(level, simNow);
      running = true;
      decisions = 0;
      lastDecisionAt = 0;
      held.length = 0;
      fireQueued = false;
      $("k-overlay").hidden = true;
      $("tank-toggle").disabled = false;
      $("tank-new").textContent = "重新开始";
      $("k-s-dec").textContent = "0";
      $("k-s-inf").innerHTML = "–<small> ms</small>";
      $("k-s-gap").innerHTML = "–<small> ms</small>";
      $("k-note").textContent = "–";
      initRows();
      updateHud();
      setPaused(false);
      refreshInfo(true);
    }

    for (const b of document.querySelectorAll("#k-level button")) {
      b.onclick = () => {
        level = b.dataset.level;
        for (const o of document.querySelectorAll("#k-level button")) o.classList.toggle("active", o === b);
        if (game) game.level = level;
        if (!running) kick();
      };
    }
    $("tank-new").onclick = start;
    $("tank-toggle").onclick = () => setPaused(!paused);
    $("nav-tank").addEventListener("click", () => {
      lastFrame = 0;
      kick();
    });

    document.addEventListener("keydown", (e) => {
      if ($("view-tank").hidden || e.target.matches("textarea, input, select")) return;
      const k = KEYMAP[e.code];
      if (!k) return;
      if (k === "space" || k === "fire") {
        fireQueued = true;
      } else if (KEY_DIR[k]) {
        if (!held.includes(k)) held.push(k);
      } else {
        return;
      }
      e.preventDefault();
    });
    document.addEventListener("keyup", (e) => {
      const k = KEYMAP[e.code];
      const i = held.indexOf(k);
      if (i >= 0) held.splice(i, 1);
    });
    window.addEventListener("blur", () => { held.length = 0; });

    initRows();
    updateHud();
  })();
}
