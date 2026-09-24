"use strict";
const $ = (id) => document.getElementById(id);
const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"];
const DIR_LABEL = { UP: "↑ 上", DOWN: "↓ 下", LEFT: "← 左", RIGHT: "→ 右" };

const state = {
  models: [],
  current: null,
  snake: null, // {session, playing, gen, lastTick}
};

const pct = (v) => (v * 100).toFixed(1) + "%";

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.status + " " + res.statusText);
  return data;
}

/* ---------- header + checkpoints ---------- */

async function refreshInfo(silent) {
  try {
    const info = await api("/api/info");
    state.models = info.models;
    if (!state.current) state.current = info.default_model;
    const ready = info.models.find((m) => m.name === state.current && m.status === "ready");
    $("engine").innerHTML = ready
      ? `<b>${ready.model_name}</b> 已加载 · ${ready.provider === "AXCLRTExecutionProvider" ? "AXCL 卡 " + ready.device_id : "AX8850 片上"} · v${info.version}`
      : `laya-axera ${info.version}`;
    renderChips();
  } catch (err) {
    if (!silent) $("engine").textContent = "服务不可达：" + err.message;
  }
}

function renderChips() {
  const box = $("chips");
  box.innerHTML = "";
  for (const m of state.models) {
    const b = document.createElement("button");
    b.className = "chip " + m.status + (m.name === state.current ? " active" : "");
    b.innerHTML = `<span class="dot"></span>${m.name}`;
    b.title = { ready: "已加载", loading: "加载中…", unloaded: "用到时再加载", error: m.error || "加载失败" }[m.status] || "";
    b.onclick = () => { state.current = m.name; renderChips(); refreshInfo(true); };
    box.appendChild(b);
  }
}

/* ---------- decisions view ---------- */

async function loadSample() {
  hide("decide-error");
  try {
    const req = await api(`/api/samples/${state.current}`);
    $("state").value = typeof req.state === "string" ? req.state : JSON.stringify(req.state, null, 2);
    $("questions").value = JSON.stringify(req.questions, null, 2);
  } catch (err) {
    showError("decide-error", err.message);
  }
}

function parseState(text) {
  const t = text.trim();
  if (t.startsWith("{") || t.startsWith("[")) {
    try { return JSON.parse(t); } catch { return text; }
  }
  return text;
}

async function runPredict() {
  hide("decide-error");
  let questions;
  try {
    questions = JSON.parse($("questions").value);
  } catch (err) {
    return showError("decide-error", "questions 不是合法 JSON：" + err.message);
  }
  const btn = $("run");
  btn.disabled = true;
  const loaded = state.models.some((m) => m.name === state.current && m.status === "ready");
  btn.textContent = loaded ? "运行中…" : "加载模型…";
  try {
    const result = await api("/api/predict", {
      model: state.current,
      state: parseState($("state").value),
      questions,
    });
    renderResults(result, questions);
  } catch (err) {
    showError("decide-error", err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "运行";
    refreshInfo(true);
  }
}

function barRow(label, value, chosen) {
  const row = document.createElement("div");
  row.className = "bar-row" + (chosen ? " chosen" : "");
  row.innerHTML = `<span class="opt" title="${esc(label)}">${chosen ? "✓ " : ""}${esc(label)}</span>
    <span class="track"><span class="fill"></span></span>
    <span class="val">${pct(value)}</span>`;
  requestAnimationFrame(() => { row.querySelector(".fill").style.width = (value * 100).toFixed(2) + "%"; });
  return row;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderResults(result, questions) {
  const box = $("results");
  box.innerHTML = "";
  for (const [name, a] of Object.entries(result.answers)) {
    const q = questions[name] || {};
    const card = document.createElement("div");
    card.className = "answer";
    card.innerHTML = `<h3>${esc(name)}<span class="qtype">${a.type}</span></h3>
      <p class="instructions">${esc(q.instructions || "")}</p>`;
    if (a.type === "choice") {
      for (const [label, p] of Object.entries(a.probabilities)) {
        card.appendChild(barRow(label, p, label === a.choice));
      }
    } else if (a.type === "score") {
      const levels = Object.keys(a.probabilities).length;
      const line = document.createElement("div");
      line.className = "score-line";
      line.innerHTML = `<span class="num">${a.score.toFixed(3)}</span><span class="of">期望分 / 最高 ${levels - 1}</span>`;
      card.appendChild(line);
      const best = Object.entries(a.probabilities).reduce((x, y) => (y[1] > x[1] ? y : x))[0];
      for (const [i, p] of Object.entries(a.probabilities)) {
        card.appendChild(barRow(`level ${i} — ${a.legend[i]}`, p, i === best));
      }
    } else {
      card.appendChild(barRow("true 成立", a.noul, a.noul >= 0.5));
      card.appendChild(barRow("false 不成立", 1 - a.noul, a.noul < 0.5));
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `置信度 ${a.confidence.toFixed(4)} · act ${a.action.act_probability.toFixed(2)} · NPU ${a.npu_latency_ms.toFixed(1)} ms`;
    card.appendChild(meta);
    box.appendChild(card);
  }
  const total = document.createElement("div");
  total.className = "total-strip";
  total.textContent = `${Object.keys(result.answers).length} 个问题，NPU 共 ${result.total_npu_latency_ms.toFixed(1)} ms，输入 ${result.usage.input_tokens} tokens`;
  box.appendChild(total);
}

/* ---------- snake view ---------- */

function initDirRows() {
  const box = $("dir-rows");
  box.innerHTML = "";
  for (const d of DIRS) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.id = "dir-" + d;
    row.innerHTML = `<span class="opt">${DIR_LABEL[d]}</span>
      <span class="track"><span class="fill"></span></span>
      <span class="val">–</span>`;
    box.appendChild(row);
  }
}

function updateDirRows(decision, executed, isManual) {
  const act = executed || (decision && decision.executed);
  for (const d of DIRS) {
    const row = $("dir-" + d);
    const p = decision ? decision.probabilities[d] : 0;
    row.classList.toggle("chosen", !!decision && act === d);
    row.querySelector(".fill").style.width = (p * 100).toFixed(2) + "%";
    row.querySelector(".val").textContent = decision ? pct(p) : "–";
    const opt = row.querySelector(".opt");
    opt.textContent = (decision && act === d ? "✓ " : "") + DIR_LABEL[d];
  }
  if (decision) pressFx("s-" + act.toLowerCase());
  const note = $("decision-note");
  if (!decision) {
    note.textContent = "–";
  } else if (isManual) {
    note.textContent = act === decision.executed
      ? `你：${DIR_LABEL[act]}，和 AI 一样`
      : `你：${DIR_LABEL[act]}　AI 会走 ${DIR_LABEL[decision.executed]}`;
  } else if (decision.intervened) {
    note.innerHTML = `<span class="flag">护栏纠正</span>：AI 想走 ${DIR_LABEL[decision.proposed]}，不安全，改为 ${DIR_LABEL[decision.executed]}`;
  } else {
    note.textContent = `AI：${DIR_LABEL[decision.executed]}`;
  }
  $("g-risk").style.width = decision ? (decision.dead_end_risk * 100).toFixed(1) + "%" : "0";
  $("g-risk-v").textContent = decision ? pct(decision.dead_end_risk) : "–";
  $("g-food").style.width = decision ? (decision.food_reachable * 100).toFixed(1) + "%" : "0";
  $("g-food-v").textContent = decision ? pct(decision.food_reachable) : "–";
}

function drawBoard(snap) {
  state._snakeSnap = snap;
  const canvas = $("board");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const cw = W / snap.width, ch = H / snap.height;
  const dark = effectiveTheme() === "dark";
  const base = GFX.token("--board");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = GFX.shade(base, dark ? 0.035 : -0.035);
  for (let y = 0; y < snap.height; y++) {
    for (let x = (y % 2); x < snap.width; x += 2) ctx.fillRect(x * cw, y * ch, cw, ch);
  }
  GFX.vignette(ctx, W, H, 0.32);

  if (snap.food) {
    const fx = (snap.food[0] + 0.5) * cw, fy = (snap.food[1] + 0.5) * ch, fr = cw * 0.3;
    GFX.floorShadow(ctx, fx + fr * 0.35, fy + fr * 0.95, fr * 1.1, fr * 0.4);
    GFX.sphere(ctx, fx, fy, fr, GFX.token("--orange"));
    ctx.fillStyle = GFX.token("--good");
    ctx.beginPath();
    ctx.ellipse(fx + fr * 0.35, fy - fr * 1.02, fr * 0.36, fr * 0.16, -0.6, 0, Math.PI * 2);
    ctx.fill();
  }

  const body = snap.body;
  const n = body.length;
  const blue = GFX.token("--blue");
  const pad = cw * 0.11;
  const colorAt = (i) => GFX.shade(blue, -0.34 * (i / Math.max(1, n - 1)));
  const span = (a, b) => {
    const x = Math.min(a[0], b[0]) * cw + pad, y = Math.min(a[1], b[1]) * ch + pad;
    return [x, y, (Math.abs(a[0] - b[0]) + 1) * cw - 2 * pad, (Math.abs(a[1] - b[1]) + 1) * ch - 2 * pad];
  };
  // shadows first so no segment casts onto its neighbour
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  for (const [x, y] of body) {
    GFX.rr(ctx, x * cw + pad + 3, y * ch + pad + 6, cw - 2 * pad, ch - 2 * pad, cw * 0.28);
    ctx.fill();
  }
  // tail to head: each link spans two tiles so the body reads as one tube
  for (let i = n - 1; i >= 1; i--) {
    const [x, y, w, h] = span(body[i], body[i - 1]);
    GFX.block(ctx, x, y, w, h, colorAt(i), { radius: cw * 0.28, shadow: false });
  }
  const [hx, hy] = body[0];
  const grow = cw * 0.04;
  GFX.block(ctx, hx * cw + pad - grow, hy * ch + pad - grow, cw - 2 * pad + 2 * grow, ch - 2 * pad + 2 * grow,
    GFX.shade(blue, 0.1), { radius: cw * 0.3, shadow: false });
  const dx = n > 1 ? Math.sign(body[0][0] - body[1][0]) : 1;
  const dy = n > 1 ? Math.sign(body[0][1] - body[1][1]) : 0;
  const cx = (hx + 0.5) * cw, cy = (hy + 0.5) * ch - cw * 0.06;
  for (const side of [-1, 1]) {
    const ex = cx + dx * cw * 0.14 - dy * side * cw * 0.17;
    const ey = cy + dy * ch * 0.14 + dx * side * ch * 0.17;
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(ex, ey, cw * 0.1, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#101418";
    ctx.beginPath(); ctx.arc(ex + dx * cw * 0.035, ey + dy * cw * 0.035, cw * 0.05, 0, Math.PI * 2); ctx.fill();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fill();
}

function updateStats(snap, stats, decision) {
  $("s-score").textContent = snap.score;
  $("s-moves").textContent = stats ? stats.moves : 0;
  $("s-int").textContent = stats ? (manual.snake ? stats.matches || 0 : stats.interventions) : 0;
  $("s-len").textContent = snap.length;
  if (decision) $("s-inf").innerHTML = decision.inference_ms.toFixed(0) + "<small> ms</small>";
  if (state.snake && state.snake.lastTick) {
    const dt = (performance.now() - state.snake.lastTick) / 1000;
    if (dt > 0) $("s-mps").innerHTML = (1 / dt).toFixed(1) + "<small> 步/s</small>";
  }
}

async function snakeNew() {
  hide("snake-error");
  $("snake-new").disabled = true;
  $("snake-new").textContent = "加载中…";
  try {
    const data = await api("/api/snake/new", {
      model: state.current,
      guarded: $("guarded").checked,
      prompt: $("prompt-style").value,
    });
    state.snake = { session: data.session, playing: true, gen: (state.snake ? state.snake.gen : 0) + 1, lastTick: 0 };
    $("overlay").hidden = true;
    $("snake-toggle").disabled = false;
    $("snake-toggle").textContent = "暂停";
    input.snakeQueue.length = 0;
    initDirRows();
    updateDirRows(null);
    drawBoard(data.state);
    updateStats(data.state, null, null);
    refreshInfo(true);
    loop(state.snake.gen);
  } catch (err) {
    showError("snake-error", err.message);
  } finally {
    $("snake-new").disabled = false;
    $("snake-new").textContent = "新开一局";
  }
}

async function loop(gen) {
  const s = state.snake;
  if (!s || s.gen !== gen || !s.playing) return;
  const speed = Number($("speed").value);
  const target = speed >= 11 ? 0 : 1000 / speed;
  const t0 = performance.now();
  let data;
  try {
    data = await api("/api/snake/step", { session: s.session, action: manualAction("snake") });
  } catch (err) {
    showError("snake-error", err.message);
    s.playing = false;
    $("snake-toggle").disabled = true;
    return;
  }
  if (!state.snake || state.snake.gen !== gen) return;
  if (data.decision) updateDirRows(data.decision, data.executed, data.manual);
  drawBoard(data.state);
  updateStats(data.state, data.stats, data.decision);
  s.lastTick = t0;
  if (data.done) {
    s.playing = false;
    $("snake-toggle").disabled = true;
    const t = $("overlay-title"), sub = $("overlay-sub");
    if (data.state.won) { t.textContent = "填满棋盘，获胜"; sub.textContent = `得分 ${data.state.score}`; }
    else {
      const reason = { wall: "撞墙", body: "撞到蛇身", reverse: "反向移动" }[data.state.death_reason] || data.state.death_reason;
      t.textContent = "本局结束：" + reason;
      sub.textContent = `得分 ${data.state.score}，共 ${data.state.ticks} 步。`;
    }
    $("overlay").hidden = false;
    return;
  }
  const elapsed = performance.now() - t0;
  setTimeout(() => loop(gen), Math.max(0, target - elapsed));
}

function snakeToggle() {
  const s = state.snake;
  if (!s) return;
  s.playing = !s.playing;
  $("snake-toggle").textContent = s.playing ? "暂停" : "继续";
  if (s.playing) loop(s.gen);
}

/* ---------- bird view ---------- */

const FDIR_LABEL = { up: "↑ 拍翅", down: "↓ 滑翔" };
let wingPhase = 0;

function initBirdRows() {
  const box = $("f-rows");
  box.innerHTML = "";
  for (const d of ["up", "down"]) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.id = "fdir-" + d;
    row.innerHTML = `<span class="opt">${FDIR_LABEL[d]}</span>
      <span class="track"><span class="fill"></span></span>
      <span class="val">–</span>`;
    box.appendChild(row);
  }
}

function updateBirdPanel(decision, executed, isManual) {
  const act = executed || (decision && decision.executed);
  for (const d of ["up", "down"]) {
    const row = $("fdir-" + d);
    const p = decision ? decision.probabilities[d] : 0;
    row.classList.toggle("chosen", !!decision && act === d);
    row.querySelector(".fill").style.width = (p * 100).toFixed(2) + "%";
    row.querySelector(".val").textContent = decision ? pct(p) : "–";
    row.querySelector(".opt").textContent = (decision && act === d ? "✓ " : "") + FDIR_LABEL[d];
  }
  if (decision) {
    pressFx("f-" + act);
    state._birdFlapping = act === "up";
  }
  const verb = (a) => (a === "up" ? "拍翅" : "滑翔");
  const note = $("f-note");
  if (!decision) {
    note.textContent = "–";
  } else if (isManual) {
    note.textContent = act === decision.executed ? `你：${verb(act)}，和 AI 一样` : `你：${verb(act)}　AI 会${verb(decision.executed)}`;
  } else if (decision.intervened) {
    note.innerHTML = `<span class="flag">护栏纠正</span>：AI 想${verb(decision.proposed)}，会撞上，改为${verb(decision.executed)}`;
  } else {
    note.textContent = `AI：${verb(decision.executed)}`;
  }
  if (decision) {
    const off = decision.gap_offset;
    $("f-g-off").style.width = Math.min(100, Math.abs(off) / 5 * 100).toFixed(1) + "%";
    $("f-g-off-v").textContent = Math.abs(off).toFixed(1) + " 格" + (off > 0 ? "（偏下）" : "（偏上）");
    const dist = Math.max(0, decision.pipe_distance);
    $("f-g-dist").style.width = Math.min(100, (dist / 9) * 100).toFixed(1) + "%";
    $("f-g-dist-v").textContent = dist < 0.05 ? "正在穿过" : dist.toFixed(1) + " 格";
  }
}

const BIRD_SCENE = {
  light: { skyTop: "#8ccbf2", skyBot: "#e6f4fc", far: "#a9cbd6", near: "#8dbb9c", ground: "#c7b186", stone: "#9aa5b8" },
  dark: { skyTop: "#070d24", skyBot: "#20345e", far: "#1a2a4b", near: "#20385a", ground: "#2d2b36", stone: "#6d7a92" },
};

function hills(ctx, W, H, baseY, amp, period, offset, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(0, H);
  for (let x = 0; x <= W; x += 12) {
    const t = (x + offset) / period;
    ctx.lineTo(x, baseY - amp * (0.55 * Math.sin(t) + 0.3 * Math.sin(t * 2.3 + 1.7) + 0.15 * Math.sin(t * 5.1)));
  }
  ctx.lineTo(W, H);
  ctx.closePath();
  ctx.fill();
}

function drawBird(snap, frame) {
  state._birdSnap = snap;
  state._birdFrame = frame || null;
  const birdY = frame ? frame[0] : snap.y;
  const shift = frame ? frame[1] : 0;
  const canvas = $("fboard");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const cw = W / snap.width, ch = H / snap.height;
  const dark = effectiveTheme() === "dark";
  const sc = dark ? BIRD_SCENE.dark : BIRD_SCENE.light;
  const scroll = ((snap.scroll || 0) - shift) * cw;

  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, sc.skyTop);
  sky.addColorStop(1, sc.skyBot);
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);
  if (dark) {
    for (let i = 0; i < 60; i++) {
      const sx = (((i * 977) % W) - scroll * 0.05 + W * 4) % W;
      const sy = (i * 431) % (H * 0.55);
      ctx.fillStyle = `rgba(255,255,255,${0.25 + ((i * 37) % 60) / 100})`;
      ctx.fillRect(sx, sy, 2, 2);
    }
  } else {
    GFX.glow(ctx, W * 0.82, H * 0.16, cw * 2.4, "#fff6c8", 0.9);
  }
  hills(ctx, W, H, H * 0.72, ch * 1.6, 210, scroll * 0.15, sc.far);
  hills(ctx, W, H, H * 0.86, ch * 1.1, 140, scroll * 0.4 + 60, sc.near);

  const groundY = H - ch * 0.34;
  const pw = snap.pipe_w * cw;
  for (const p of snap.pipes) {
    const x = (p.x + shift) * cw - pw / 2;
    const g = p.gap != null ? p.gap : snap.gap;
    const topEnd = (p.gap_y - g / 2) * ch, botStart = (p.gap_y + g / 2) * ch;
    GFX.floorShadow(ctx, x + pw * 0.6, groundY + ch * 0.12, pw * 0.9, ch * 0.16, 0.3);
    GFX.cylinder(ctx, x, -4, pw, topEnd + 4, sc.stone);
    GFX.cylinder(ctx, x, botStart, pw, groundY - botStart, sc.stone);
    ctx.fillStyle = "rgba(0,0,0,0.16)";
    for (let y = topEnd - ch * 1.4; y > 0; y -= ch * 1.4) ctx.fillRect(x, y, pw, 2);
    for (let y = botStart + ch * 1.4; y < groundY; y += ch * 1.4) ctx.fillRect(x, y, pw, 2);
    const capW = pw * 1.16, capH = ch * 0.34, capX = x - (capW - pw) / 2;
    GFX.block(ctx, capX, topEnd - capH, capW, capH, GFX.shade(sc.stone, 0.08), { depth: 4, radius: 3, shadow: false });
    GFX.block(ctx, capX, botStart, capW, capH, GFX.shade(sc.stone, 0.08), { depth: 4, radius: 3, shadow: false });
  }

  ctx.fillStyle = sc.ground;
  ctx.fillRect(0, groundY, W, H - groundY);
  ctx.fillStyle = GFX.shade(sc.ground, -0.12);
  for (let x = -((scroll % 36) + 36); x < W; x += 36) ctx.fillRect(x, groundY + 3, 18, H - groundY - 3);
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.fillRect(0, groundY, W, 2);

  const bx = snap.bird_x * cw, by = birdY * ch, br = snap.bird_r * cw * 1.35;
  const vy = frame && state._birdPrevY != null ? birdY - state._birdPrevY : snap.vy || 0;
  state._birdPrevY = birdY;
  const altitude = GFX.clamp((groundY - by) / H, 0, 1);
  GFX.floorShadow(ctx, bx, groundY + 2, br * (1.3 - altitude * 0.6), br * 0.3, 0.35 * (1 - altitude * 0.7));

  ctx.save();
  ctx.translate(bx, by);
  ctx.rotate(GFX.clamp(vy * 3.4, -0.5, 0.8));
  const orange = GFX.token("--orange");
  GFX.sphere(ctx, 0, 0, br, orange);
  ctx.fillStyle = "rgba(255,244,214,0.55)";
  ctx.beginPath(); ctx.ellipse(br * 0.18, br * 0.42, br * 0.52, br * 0.34, -0.2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#f2b33d";
  ctx.beginPath();
  ctx.moveTo(br * 0.82, -br * 0.08); ctx.lineTo(br * 1.42, br * 0.1); ctx.lineTo(br * 0.8, br * 0.3);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.arc(br * 0.42, -br * 0.34, br * 0.26, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#15181d";
  ctx.beginPath(); ctx.arc(br * 0.5, -br * 0.32, br * 0.13, 0, Math.PI * 2); ctx.fill();
  // The wing beats fast on a flap step and idles while gliding; raised shows its
  // back, lowered shows its underside, so the two faces read as different colors.
  wingPhase += state._birdFlapping ? 0.62 : 0.17;
  const beat = Math.sin(wingPhase);
  ctx.translate(-br * 0.22, -beat * br * 0.42);
  ctx.rotate(-beat * 0.38);
  ctx.fillStyle = GFX.token(beat >= 0 ? "--wing-back" : "--wing-front");
  ctx.beginPath();
  ctx.ellipse(-br * 0.2, 0, br * 0.78, Math.max(br * 0.14, br * 0.46 * Math.abs(beat)), 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function updateBirdStats(snap, stats, decision) {
  $("f-s-score").textContent = snap.score;
  $("f-s-steps").textContent = stats ? stats.steps : 0;
  $("f-s-int").textContent = stats ? (manual.bird ? stats.matches || 0 : stats.interventions) : 0;
  if (decision) $("f-s-inf").innerHTML = decision.inference_ms.toFixed(0) + "<small> ms</small>";
  if (stats && stats.steps > 0) $("f-s-flap").textContent = pct(stats.flaps / stats.steps);
  const f = state.bird;
  if (f && f.lastTick) {
    const dt = (performance.now() - f.lastTick) / 1000;
    if (dt > 0) $("f-s-mps").innerHTML = (1 / dt).toFixed(1) + "<small> 步/s</small>";
  }
}

async function birdNew() {
  hide("bird-error");
  $("bird-new").disabled = true;
  $("bird-new").textContent = "加载中…";
  try {
    const data = await api("/api/bird/new", {
      model: state.current,
      guarded: $("f-guarded").checked,
    });
    state.bird = { session: data.session, playing: true, gen: (state.bird ? state.bird.gen : 0) + 1, lastTick: 0 };
    $("f-overlay").hidden = true;
    $("bird-toggle").disabled = false;
    $("bird-toggle").textContent = "暂停";
    input.birdFlap = false;
    state._birdPrevY = null;
    initBirdRows();
    updateBirdPanel(null);
    drawBird(data.state, null);
    updateBirdStats(data.state, null, null);
    refreshInfo(true);
    birdLoop(state.bird.gen);
  } catch (err) {
    showError("bird-error", err.message);
  } finally {
    $("bird-new").disabled = false;
    $("bird-new").textContent = "新开一局";
  }
}

async function birdLoop(gen) {
  const f = state.bird;
  if (!f || f.gen !== gen || !f.playing) return;
  const speed = Number($("f-speed").value);
  const target = speed >= 11 ? 0 : 1000 / speed;
  const t0 = performance.now();
  let data;
  try {
    data = await api("/api/bird/step", { session: f.session, action: manualAction("bird") });
  } catch (err) {
    showError("bird-error", err.message);
    f.playing = false;
    $("bird-toggle").disabled = true;
    return;
  }
  if (!state.bird || state.bird.gen !== gen) return;
  if (data.decision) updateBirdPanel(data.decision, data.executed, data.manual);
  updateBirdStats(data.state, data.stats, data.decision);
  const remain = Math.max(0, target - (performance.now() - t0));
  if (data.trail && data.trail.length > 1 && remain >= 40) {
    await playTrail(data.trail, remain, (fr) => drawBird(data.state, fr));
    if (!state.bird || state.bird.gen !== gen) return;
  }
  drawBird(data.state, null);
  f.lastTick = t0;
  if (data.done) {
    f.playing = false;
    $("bird-toggle").disabled = true;
    const reason = { pipe: "撞上石柱", floor: "落地" }[data.state.death_reason] || data.state.death_reason;
    $("f-overlay-title").textContent = "本局结束：" + reason;
    $("f-overlay-sub").textContent = `得分 ${data.state.score}，共 ${data.state.steps} 次决策。`;
    $("f-overlay").hidden = false;
    return;
  }
  const elapsed = performance.now() - t0;
  setTimeout(() => birdLoop(gen), Math.max(0, target - elapsed));
}

function birdToggle() {
  const f = state.bird;
  if (!f) return;
  f.playing = !f.playing;
  $("bird-toggle").textContent = f.playing ? "暂停" : "继续";
  if (f.playing) birdLoop(f.gen);
}

/* ---------- bricks view ---------- */

const B_ACTS = ["left", "hold", "right"];
const B_LABEL = { left: "◀ 左移", hold: "■ 不动", right: "▶ 右移" };
const B_BTN = { left: "b-left", hold: "b-hold", right: "b-right" };

function initBricksRows() {
  const box = $("b-rows");
  box.innerHTML = "";
  for (const a of B_ACTS) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.id = "bact-" + a;
    row.innerHTML = `<span class="opt">${B_LABEL[a]}</span>
      <span class="track"><span class="fill"></span></span>
      <span class="val">–</span>`;
    box.appendChild(row);
  }
}

function updateBricksPanel(decision, snap, executed, isManual) {
  const act = executed || (decision && decision.executed);
  for (const a of B_ACTS) {
    const row = $("bact-" + a);
    const p = decision ? decision.probabilities[a] : 0;
    row.classList.toggle("chosen", !!decision && act === a);
    row.querySelector(".fill").style.width = (p * 100).toFixed(2) + "%";
    row.querySelector(".val").textContent = decision ? pct(p) : "–";
    row.querySelector(".opt").textContent = (decision && act === a ? "✓ " : "") + B_LABEL[a];
  }
  const note = $("b-note");
  if (!decision) {
    note.textContent = "–";
  } else if (isManual) {
    note.textContent = act === decision.executed ? `你：${B_LABEL[act]}，和 AI 一样` : `你：${B_LABEL[act]}　AI 会选 ${B_LABEL[decision.executed]}`;
  } else if (decision.intervened) {
    note.innerHTML = `<span class="flag">护栏纠正</span>：AI 想${B_LABEL[decision.proposed]}，会漏球，改为${B_LABEL[decision.executed]}`;
  } else {
    note.textContent = `AI：${B_LABEL[decision.executed]}`;
  }
  if (decision) {
    pressFx(B_BTN[act]);
    $("b-g-err").style.width = Math.min(100, (decision.error / 6) * 100).toFixed(1) + "%";
    $("b-g-err-v").textContent = decision.error.toFixed(2) + " 格";
  }
  if (snap) {
    const total = snap.bricks_total || snap.bricks_left || 1;
    $("b-g-brick").style.width = ((snap.bricks_left / total) * 100).toFixed(1) + "%";
    $("b-g-brick-v").textContent = snap.bricks_left + " / " + total;
  }
}

const bricksFx = new GFX.Particles();

function drawBricks(snap, frame) {
  const prev = state._bricksSnap;
  state._bricksSnap = snap;
  state._bricksFrame = frame || null;
  const ballX = frame ? frame[0] : snap.ball[0];
  const ballY = frame ? frame[1] : snap.ball[1];
  const padX = frame ? frame[2] : snap.paddle_x;
  const canvas = $("bboard");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const cw = W / snap.width, ch = H / snap.height;
  const colors = T_COLORS[effectiveTheme()];
  const rowColor = (r) => colors[(((r - 1) % colors.length) + colors.length) % colors.length];

  if (prev && prev !== snap && prev.bricks.length > snap.bricks.length) {
    const alive = new Set(snap.bricks.map(([r, c]) => r * 1000 + c));
    for (const [r, c] of prev.bricks) {
      if (!alive.has(r * 1000 + c)) {
        bricksFx.burst((c + 0.5) * snap.brick_w * cw, (r + 0.5) * ch, rowColor(r), 12, 260, cw * 0.18);
      }
    }
  }

  GFX.floor(ctx, W, H, GFX.token("--board"));
  for (const [r, c] of snap.bricks) {
    GFX.block(ctx, c * snap.brick_w * cw + 2, r * ch + 2, snap.brick_w * cw - 4, ch - 4, rowColor(r),
      { depth: Math.round(ch * 0.16), radius: 5 });
  }

  const blue = GFX.token("--blue");
  const px = (padX - snap.paddle_w / 2) * cw, pwid = snap.paddle_w * cw, py = snap.paddle_y * ch;
  GFX.glow(ctx, padX * cw, py + ch * 0.3, pwid * 0.75, blue, 0.28);
  GFX.capsule(ctx, px, py, pwid, ch * 0.5, blue);

  const trail = (state._bricksBallTrail = state._bricksBallTrail || []);
  trail.push([ballX, ballY]);
  if (trail.length > 9) trail.shift();
  const orange = GFX.token("--orange");
  const br = snap.ball_r * cw * 1.3;
  trail.forEach(([x, y], i) => {
    ctx.fillStyle = GFX.alpha(orange, (0.3 * (i + 1)) / trail.length);
    ctx.beginPath();
    ctx.arc(x * cw, y * ch, br * (0.45 + (0.55 * i) / trail.length), 0, Math.PI * 2);
    ctx.fill();
  });
  GFX.glow(ctx, ballX * cw, ballY * ch, br * 3, orange, 0.35);
  GFX.sphere(ctx, ballX * cw, ballY * ch, br, orange);
  bricksFx.draw(ctx);
}

function updateBricksStats(snap, stats, decision) {
  $("b-s-score").textContent = snap.score;
  $("b-s-steps").textContent = stats ? stats.steps : 0;
  $("b-s-lives").textContent = snap.lives;
  $("b-s-int").textContent = stats ? (manual.bricks ? stats.matches || 0 : stats.interventions) : 0;
  if (decision) $("b-s-inf").innerHTML = decision.inference_ms.toFixed(0) + "<small> ms</small>";
  const b = state.bricks;
  if (b && b.lastTick) {
    const dt = (performance.now() - b.lastTick) / 1000;
    if (dt > 0) $("b-s-sps").innerHTML = (1 / dt).toFixed(1) + "<small> 步/s</small>";
  }
}

async function bricksNew() {
  hide("bricks-error");
  $("bricks-new").disabled = true;
  $("bricks-new").textContent = "加载中…";
  try {
    const data = await api("/api/bricks/new", {
      model: state.current,
      guarded: $("b-guarded").checked,
    });
    state.bricks = { session: data.session, playing: true, gen: (state.bricks ? state.bricks.gen : 0) + 1, lastTick: 0 };
    $("b-overlay").hidden = true;
    $("bricks-toggle").disabled = false;
    $("bricks-toggle").textContent = "暂停";
    state._bricksSnap = null;
    state._bricksBallTrail = [];
    initBricksRows();
    updateBricksPanel(null, data.state);
    drawBricks(data.state, null);
    updateBricksStats(data.state, null, null);
    refreshInfo(true);
    bricksLoop(state.bricks.gen);
  } catch (err) {
    showError("bricks-error", err.message);
  } finally {
    $("bricks-new").disabled = false;
    $("bricks-new").textContent = "新开一局";
  }
}

async function bricksLoop(gen) {
  const b = state.bricks;
  if (!b || b.gen !== gen || !b.playing) return;
  const speed = Number($("b-speed").value);
  const target = speed >= 11 ? 0 : 1000 / speed;
  const t0 = performance.now();
  let data;
  try {
    data = await api("/api/bricks/step", { session: b.session, action: manualAction("bricks") });
  } catch (err) {
    showError("bricks-error", err.message);
    b.playing = false;
    $("bricks-toggle").disabled = true;
    return;
  }
  if (!state.bricks || state.bricks.gen !== gen) return;
  if (data.decision) updateBricksPanel(data.decision, data.state, data.executed, data.manual);
  updateBricksStats(data.state, data.stats, data.decision);
  const remain = Math.max(0, target - (performance.now() - t0));
  if (data.trail && data.trail.length > 1 && remain >= 40) {
    await playTrail(data.trail, remain, (f) => drawBricks(data.state, f));
    if (!state.bricks || state.bricks.gen !== gen) return;
  }
  drawBricks(data.state, null);
  b.lastTick = t0;
  if (data.done) {
    b.playing = false;
    $("bricks-toggle").disabled = true;
    const won = data.state.won;
    $("b-overlay-title").textContent = won ? "全部砖块清空，胜利" : "本局结束：球漏掉了";
    $("b-overlay-sub").textContent = `得分 ${data.state.score}，共 ${data.state.steps} 次决策。`;
    $("b-overlay").hidden = false;
    return;
  }
  const elapsed = performance.now() - t0;
  setTimeout(() => bricksLoop(gen), Math.max(0, target - elapsed));
}

function bricksToggle() {
  const b = state.bricks;
  if (!b) return;
  b.playing = !b.playing;
  $("bricks-toggle").textContent = b.playing ? "暂停" : "继续";
  if (b.playing) bricksLoop(b.gen);
}

/* ---------- blocks view ---------- */

const T_MARKS = ["\u2460", "\u2461", "\u2462", "\u2463"];
const T_COLORS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"],
};

function candText(c) {
  const lines = c.lines ? `消${c.lines}行` : "不消行";
  const holes = c.holes ? `埋${c.holes}洞` : "无洞";
  const height = c.height > 15 ? "近顶" : c.height > 10 ? "变高" : "低位";
  return `${lines}·${holes}·${height}`;
}

function buildBlocksPanel(cands) {
  const box = $("t-cands");
  box.innerHTML = "";
  cands.forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "t-cand";
    row.innerHTML = `<span class="opt" title="${esc(c.statement)}">${T_MARKS[i]} ${candText(c)}</span>
      <span class="track"><span class="fill"></span></span>
      <span class="val">…</span>`;
    box.appendChild(row);
  });
}

function pressFx(id) {
  const b = $(id);
  b.classList.add("pressed");
  setTimeout(() => b.classList.remove("pressed"), 150);
}

function pulseRow(i) {
  const row = $("t-cands").children[i];
  if (!row) return;
  row.classList.add("pulse");
  setTimeout(() => row.classList.remove("pulse"), 280);
}

function revealBlocksCand(i, c) {
  const row = $("t-cands").children[i];
  if (!row) return;
  row.querySelector(".fill").style.width = (c.p_good * 100).toFixed(1) + "%";
  row.querySelector(".val").textContent = pct(c.p_good);
}

function finalizeBlocksPanel(decision) {
  const row = $("t-cands").children[decision.executed];
  if (row) {
    row.classList.add("chosen");
    const opt = row.querySelector(".opt");
    opt.textContent = "✓ " + opt.textContent;
  }
  const note = $("t-note");
  if (decision.intervened) {
    note.innerHTML = `<span class="flag">护栏纠正</span>：${T_MARKS[decision.proposed]} 会堆得太高，改用 ${T_MARKS[decision.executed]}`;
  } else if (decision.executed === decision.heuristic_best) {
    note.textContent = `AI 选 ${T_MARKS[decision.executed]}`;
  } else {
    note.textContent = `AI 选 ${T_MARKS[decision.executed]}（和启发式排第一的不同）`;
  }
}

function updateBlocksPanel(decision) {
  if (!decision) { $("t-cands").innerHTML = ""; return; }
  buildBlocksPanel(decision.candidates);
  decision.candidates.forEach((c, i) => revealBlocksCand(i, c));
  finalizeBlocksPanel(decision);
}

const T_SHAPE_IDX = { I: 0, O: 1, T: 2, S: 3, Z: 4, J: 5, L: 6 };

function renderBlocksFrame(snap, opts) {
  // opts: {piece: {cells, colorIdx, dyPx}, outlineCells, flashRows, ghosts}
  const canvas = $("tboard");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const cw = W / snap.width, ch = H / snap.height;
  const colors = T_COLORS[effectiveTheme()];
  const ink = GFX.token("--ink");
  const blue = GFX.token("--blue");
  GFX.floor(ctx, W, H, GFX.token("--board"));
  ctx.strokeStyle = "rgba(0,0,0,0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 1; x < snap.width; x++) { ctx.moveTo(x * cw + 0.5, 0); ctx.lineTo(x * cw + 0.5, H); }
  for (let y = 1; y < snap.height; y++) { ctx.moveTo(0, y * ch + 0.5); ctx.lineTo(W, y * ch + 0.5); }
  ctx.stroke();
  const cube = (x, py, color, shadow) =>
    GFX.block(ctx, x * cw + 1, py + 1, cw - 2, ch - 2, color, { depth: 4, radius: 4, shadow });
  for (let y = 0; y < snap.height; y++) {
    for (let x = 0; x < snap.width; x++) {
      const v = snap.board[y][x];
      if (v) cube(x, y * ch, colors[(v - 1) % colors.length], false);
    }
  }
  if (opts && opts.ghosts) {
    ctx.save();
    ctx.font = "600 13px system-ui, sans-serif";
    for (const g of opts.ghosts) {
      const pulseAge = g.pulseT ? performance.now() - g.pulseT : 1e9;
      if (pulseAge < 240) {
        ctx.fillStyle = GFX.alpha(blue, 0.5 * (1 - pulseAge / 240));
        for (const [x, y] of g.cells) ctx.fillRect(x * cw + 1, y * ch + 1, cw - 2, ch - 2);
      }
      ctx.setLineDash(g.chosen ? [] : [5, 4]);
      ctx.lineWidth = g.chosen ? 2.5 : 1.5;
      ctx.strokeStyle = g.chosen ? blue : ink;
      ctx.globalAlpha = g.revealed ? 0.95 : 0.35;
      if (g.chosen) {
        ctx.fillStyle = GFX.alpha(blue, 0.25);
        for (const [x, y] of g.cells) ctx.fillRect(x * cw + 1, y * ch + 1, cw - 2, ch - 2);
      }
      for (const [x, y] of g.cells) {
        ctx.beginPath();
        ctx.roundRect(x * cw + 1.5, y * ch + 1.5, cw - 3, ch - 3, 3);
        ctx.stroke();
      }
      const tx = Math.min(...g.cells.map((c) => c[0]));
      const ty = Math.min(...g.cells.filter((c) => c[0] === tx).map((c) => c[1]));
      ctx.globalAlpha = 1;
      ctx.fillStyle = g.chosen ? blue : ink;
      ctx.fillText(g.mark, tx * cw + 4, ty * ch + 16);
    }
    ctx.restore();
  }
  if (opts && opts.piece) {
    const color = colors[opts.piece.colorIdx % colors.length];
    for (const [x, y] of opts.piece.cells) {
      const py = y * ch + opts.piece.dyPx;
      if (py + ch >= 0) cube(x, py, color, true);
    }
  }
  if (opts && opts.flashRows) {
    ctx.fillStyle = GFX.alpha(ink, 0.75);
    for (const y of opts.flashRows) ctx.fillRect(0, y * ch, W, ch);
  }
  if (opts && opts.outlineCells) {
    ctx.strokeStyle = ink;
    ctx.lineWidth = 2;
    for (const [x, y] of opts.outlineCells) {
      ctx.beginPath();
      ctx.roundRect(x * cw + 1.5, y * ch + 1.5, cw - 3, ch - 3, 3);
      ctx.stroke();
    }
  }
}

function drawBlocks(snap, placedCells) {
  state._blocksSnap = snap;
  state._blocksPlaced = placedCells;
  renderBlocksFrame(snap, { outlineCells: placedCells });
}

// Rotation tables mirroring laya_axera/blocks/game.py exactly (indices must match).
const T_SHAPES = {
  I: [[[0,0],[1,0],[2,0],[3,0]], [[0,0],[0,1],[0,2],[0,3]]],
  O: [[[0,0],[1,0],[0,1],[1,1]]],
  T: [[[0,0],[1,0],[2,0],[1,1]], [[1,0],[0,1],[1,1],[1,2]], [[1,0],[0,1],[1,1],[2,1]], [[0,0],[0,1],[1,1],[0,2]]],
  S: [[[1,0],[2,0],[0,1],[1,1]], [[0,0],[0,1],[1,1],[1,2]]],
  Z: [[[0,0],[1,0],[1,1],[2,1]], [[1,0],[0,1],[1,1],[0,2]]],
  J: [[[0,0],[0,1],[1,1],[2,1]], [[0,0],[1,0],[0,1],[0,2]], [[0,0],[1,0],[2,0],[2,1]], [[1,0],[1,1],[0,2],[1,2]]],
  L: [[[2,0],[0,1],[1,1],[2,1]], [[0,0],[0,1],[0,2],[1,2]], [[0,0],[1,0],[2,0],[0,1]], [[0,0],[1,0],[1,1],[1,2]]],
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Replay the physics frames the server actually ran, spread over the budget.
function playTrail(trail, budgetMs, render) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    let last = -1;
    const frame = (now) => {
      const p = Math.min(1, (now - t0) / budgetMs);
      const idx = Math.min(trail.length - 1, Math.floor(p * trail.length));
      if (idx !== last) { render(trail[idx]); last = idx; }
      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

function pieceFrame(prevSnap, letter, rot, col, rowFloat, ghosts) {
  const ch = $("tboard").height / prevSnap.height;
  const cells = T_SHAPES[letter][rot].map(([x, y]) => [col + x, y]);
  renderBlocksFrame(prevSnap, {
    piece: { cells, colorIdx: T_SHAPE_IDX[letter] ?? 0, dyPx: rowFloat * ch },
    ghosts,
  });
}

async function animatePlacement(prevSnap, cand, letter, budgetMs, isStale, ghosts, startRow = 0) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const rots = T_SHAPES[letter];
  const W = prevSnap.width;
  const width = (r) => Math.max(...rots[r].map((c) => c[0])) + 1;
  const clampCol = (c, r) => Math.max(0, Math.min(c, W - width(r)));
  const spawnCol = clampCol(Math.floor((W - width(0)) / 2), 0);

  // Phase A: spawn at top center, step through rotations in place.
  const rotFrames = cand.rotation + 1;
  const rotMs = Math.max(60, (budgetMs * 0.22) / rotFrames);
  for (let r = 0; r <= cand.rotation; r++) {
    if (isStale()) return;
    if (r > 0) pressFx("t-rotate");
    pieceFrame(prevSnap, letter, r, clampCol(spawnCol, r), startRow, ghosts);
    await sleep(rotMs);
  }

  // Phase B: slide cell by cell toward the target column, drifting down a little.
  const fromCol = clampCol(spawnCol, cand.rotation);
  const dist = Math.abs(cand.col - fromCol);
  if (dist > 0) {
    const stepMs = Math.max(24, (budgetMs * 0.28) / dist);
    const dir = cand.col > fromCol ? 1 : -1;
    for (let i = 1; i <= dist; i++) {
      if (isStale()) return;
      pressFx(dir > 0 ? "t-right" : "t-left");
      pieceFrame(prevSnap, letter, cand.rotation, fromCol + dir * i, startRow + (0.8 * i) / dist, ghosts);
      await sleep(stepMs);
    }
  }

  // Phase C: accelerating drop to the resting row.
  const fallFrom = dist > 0 ? startRow + 0.8 : startRow;
  pressFx("t-drop");
  const dropMs = Math.max(140, budgetMs * 0.5);
  await new Promise((resolve) => {
    const t0 = performance.now();
    const frame = (now) => {
      if (isStale()) return resolve();
      const p = Math.min(1, (now - t0) / dropMs);
      pieceFrame(prevSnap, letter, cand.rotation, cand.col, fallFrom + (cand.row - fallFrom) * p * p, ghosts);
      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

function fullRowsAfter(prevSnap, cells) {
  const grid = prevSnap.board.map((r) => r.slice());
  for (const [x, y] of cells) grid[y][x] = 8;
  const rows = [];
  grid.forEach((r, y) => { if (r.every((v) => v)) rows.push(y); });
  return rows;
}

function updateBlocksStats(snap, stats, decision) {
  $("t-s-score").textContent = snap.score;
  $("t-s-lines").textContent = snap.lines;
  $("t-s-pieces").textContent = snap.pieces;
  $("t-s-int").textContent = stats ? stats.interventions : 0;
  if (decision) $("t-s-inf").innerHTML = decision.inference_ms.toFixed(0) + "<small> ms</small>";
  const t = state.blocks;
  if (t && t.lastTick) {
    const dt = (performance.now() - t.lastTick) / 1000;
    if (dt > 0) $("t-s-mps").innerHTML = (1 / dt).toFixed(1) + "<small> 块/s</small>";
  }
}

async function blocksNew() {
  hide("blocks-error");
  manualStop();
  $("blocks-new").disabled = true;
  $("blocks-new").textContent = "加载中…";
  try {
    const data = await api("/api/blocks/new", {
      model: state.current,
      guarded: $("t-guarded").checked,
    });
    state.blocks = { session: data.session, playing: true, gen: (state.blocks ? state.blocks.gen : 0) + 1, lastTick: 0 };
    $("t-overlay").hidden = true;
    $("blocks-toggle").disabled = false;
    $("blocks-toggle").textContent = "暂停";
    updateBlocksPanel(null);
    drawBlocks(data.state, null);
    updateBlocksStats(data.state, null, null);
    refreshInfo(true);
    if (state.blocksMode === "manual") {
      $("blocks-toggle").disabled = true;
      manualSpawn();
    } else {
      blocksLoop(state.blocks.gen);
    }
  } catch (err) {
    showError("blocks-error", err.message);
  } finally {
    $("blocks-new").disabled = false;
    $("blocks-new").textContent = "新开一局";
  }
}

async function blocksLoop(gen) {
  const t = state.blocks;
  if (!t || t.gen !== gen || !t.playing) return;
  const speed = Number($("t-speed").value);
  const target = speed >= 11 ? 0 : 1000 / speed;
  const t0 = performance.now();
  let data;
  try {
    data = await api("/api/blocks/step", { session: t.session });
  } catch (err) {
    showError("blocks-error", err.message);
    t.playing = false;
    $("blocks-toggle").disabled = true;
    return;
  }
  if (!state.blocks || state.blocks.gen !== gen) return;
  if (data.decision) {
    const dec = data.decision;
    const cand = dec.candidates[dec.executed];
    const cells = cand.cells;
    const prev = state._blocksSnap;
    if (prev && prev.board) {
      const isStale = () => !state.blocks || state.blocks.gen !== gen;
      const budget = target > 0 ? Math.max(900, Math.min(2400, target)) : 650;
      const ghosts = dec.candidates.map((c, i) => ({
        cells: c.cells, mark: T_MARKS[i], revealed: false, chosen: false,
      }));
      const spawnCol = Math.max(0, Math.floor(
        (prev.width - (Math.max(...T_SHAPES[prev.current][0].map(c => c[0])) + 1)) / 2));
      const DRIFT = 2.2; // the fresh piece keeps sinking while the model deliberates
      buildBlocksPanel(dec.candidates);
      $("t-note").textContent = "AI 打分中…";
      const showMs = budget * 0.58;
      const n = dec.candidates.length;
      const events = dec.candidates.map((_, i) => ({ at: showMs * (0.12 + (0.62 * i) / n), fired: false }));
      let winnerFired = false;
      await new Promise((resolve) => {
        const t0 = performance.now();
        const frame = (now) => {
          if (isStale()) return resolve();
          const el = now - t0;
          events.forEach((ev, i) => {
            if (!ev.fired && el >= ev.at) {
              ev.fired = true;
              ghosts[i].revealed = true;
              ghosts[i].pulseT = now;
              revealBlocksCand(i, dec.candidates[i]);
              pulseRow(i);
            }
          });
          if (!winnerFired && el >= showMs * 0.86) {
            winnerFired = true;
            ghosts[dec.executed].chosen = true;
            ghosts[dec.executed].pulseT = now;
            finalizeBlocksPanel(dec);
            pulseRow(dec.executed);
          }
          pieceFrame(prev, prev.current, 0, spawnCol, DRIFT * Math.min(1, el / showMs), ghosts);
          if (el < showMs + 240) requestAnimationFrame(frame);
          else resolve();
        };
        requestAnimationFrame(frame);
      });
      if (isStale()) return;
      // rotate, slide, drop into the winning ghost, continuing from the drift row
      await animatePlacement(prev, cand, prev.current, budget * 0.42, isStale,
        [ghosts[dec.executed]], DRIFT);
      if (isStale()) return;
      const rows = fullRowsAfter(prev, cells);
      if (rows.length) {
        renderBlocksFrame(prev, { piece: { cells, colorIdx: T_SHAPE_IDX[prev.current] ?? 0, dyPx: 0 }, flashRows: rows });
        await sleep(130);
        if (isStale()) return;
      }
    } else {
      updateBlocksPanel(dec);
    }
    drawBlocks(data.state, cells);
  } else {
    drawBlocks(data.state, null);
  }
  updateBlocksStats(data.state, data.stats, data.decision);
  t.lastTick = t0;
  if (data.done) {
    t.playing = false;
    $("blocks-toggle").disabled = true;
    $("t-overlay-title").textContent = "本局结束：堆到顶了";
    $("t-overlay-sub").textContent = `得分 ${data.state.score}，消 ${data.state.lines} 行，共 ${data.state.pieces} 块。`;
    $("t-overlay").hidden = false;
    return;
  }
  const elapsed = performance.now() - t0;
  setTimeout(() => blocksLoop(gen), Math.max(0, target - elapsed));
}

function blocksToggle() {
  const t = state.blocks;
  if (!t) return;
  t.playing = !t.playing;
  $("blocks-toggle").textContent = t.playing ? "暂停" : "继续";
  if (t.playing) blocksLoop(t.gen);
}

/* ---------- blocks manual mode ---------- */

const man = { active: false, letter: null, rot: 0, col: 0, rowFloat: 0, timer: null };
state.blocksMode = "auto";

function jsFits(board, cells, col, row) {
  for (const [x, y] of cells) {
    const bx = col + x, by = row + y;
    if (bx < 0 || bx >= board[0].length || by < 0 || by >= board.length || board[by][bx]) return false;
  }
  return true;
}

function jsDropRow(board, cells, col) {
  if (!jsFits(board, cells, col, 0)) return null;
  let r = 0;
  while (jsFits(board, cells, col, r + 1)) r++;
  return r;
}

function manualLanding(rot = man.rot, col = man.col) {
  return jsDropRow(state._blocksSnap.board, T_SHAPES[man.letter][rot], col);
}

function manualRender() {
  const land = manualLanding();
  const ghosts = land === null ? [] : [{
    cells: T_SHAPES[man.letter][man.rot].map(([x, y]) => [man.col + x, land + y]),
    mark: "◎", revealed: true, chosen: false,
  }];
  pieceFrame(state._blocksSnap, man.letter, man.rot, man.col, man.rowFloat, ghosts);
}

function manualSpawn() {
  const prev = state._blocksSnap;
  if (!prev || !prev.alive || !state.blocks) return;
  man.letter = prev.current;
  man.rot = 0;
  const w = Math.max(...T_SHAPES[man.letter][0].map((c) => c[0])) + 1;
  man.col = Math.max(0, Math.floor((prev.width - w) / 2));
  man.rowFloat = 0;
  man.active = true;
  $("t-note").textContent = "轮到你了";
  clearInterval(man.timer);
  man.timer = setInterval(manualGravity, 650);
  manualRender();
}

function manualStop() {
  man.active = false;
  clearInterval(man.timer);
}

function manualGravity() {
  if (!man.active) return;
  const land = manualLanding();
  if (land === null || man.rowFloat + 1 >= land) {
    man.rowFloat = land === null ? man.rowFloat : land;
    manualRender();
    manualLock();
  } else {
    man.rowFloat += 1;
    manualRender();
  }
}

function manualTry(rot, col) {
  if (!man.active) return;
  const cells = T_SHAPES[man.letter][rot];
  const maxX = Math.max(...cells.map((c) => c[0]));
  if (col < 0 || col > state._blocksSnap.width - 1 - maxX) return;
  const land = jsDropRow(state._blocksSnap.board, cells, col);
  if (land === null || land < man.rowFloat) return;
  man.rot = rot;
  man.col = col;
  manualRender();
}

function manualRotate() {
  if (!man.active) return;
  const rot = (man.rot + 1) % T_SHAPES[man.letter].length;
  const maxX = Math.max(...T_SHAPES[man.letter][rot].map((c) => c[0]));
  manualTry(rot, Math.min(man.col, state._blocksSnap.width - 1 - maxX));
}

async function manualLock() {
  if (!man.active) return;
  manualStop();
  const prev = state._blocksSnap;
  const fromRow = man.rowFloat;
  let data;
  try {
    data = await api("/api/blocks/place", { session: state.blocks.session, rotation: man.rot, col: man.col });
  } catch (err) {
    showError("blocks-error", err.message);
    return;
  }
  const your = data.your;
  // finish the fall from wherever the player left it
  await new Promise((resolve) => {
    const dropMs = 160;
    const t0 = performance.now();
    const frame = (now) => {
      const p = Math.min(1, (now - t0) / dropMs);
      pieceFrame(prev, man.letter, man.rot, man.col, fromRow + (your.row - fromRow) * p * p);
      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
  const rows = fullRowsAfter(prev, your.cells);
  if (rows.length) {
    renderBlocksFrame(prev, { piece: { cells: your.cells, colorIdx: T_SHAPE_IDX[man.letter] ?? 0, dyPx: 0 }, flashRows: rows });
    await sleep(130);
  }
  drawBlocks(data.state, your.cells);
  renderManualPanel(data);
  updateBlocksStats(data.state, data.stats, { inference_ms: data.inference_ms });
  $("t-s-int").textContent = data.stats.matches || 0;
  if (data.done) {
    $("t-overlay-title").textContent = "本局结束：堆到顶了";
    $("t-overlay-sub").textContent = `得分 ${data.state.score}，消 ${data.state.lines} 行，和 AI 选得一样 ${data.stats.matches || 0}/${data.state.pieces} 次。`;
    $("t-overlay").hidden = false;
    return;
  }
  if (state.blocksMode === "manual") manualSpawn();
}

function renderManualPanel(data) {
  const box = $("t-cands");
  box.innerHTML = "";
  const mkRow = (label, c, chosen, starred) => {
    const row = document.createElement("div");
    row.className = "t-cand" + (chosen ? " chosen" : "");
    row.innerHTML = `<span class="opt" title="${esc(c.statement)}">${chosen ? "✓ " : ""}${starred ? "★ " : ""}${label} ${candText(c)}</span>
      <span class="track"><span class="fill"></span></span>
      <span class="val">${pct(c.p_good)}</span>`;
    requestAnimationFrame(() => { row.querySelector(".fill").style.width = (c.p_good * 100).toFixed(1) + "%"; });
    box.appendChild(row);
  };
  mkRow("你", data.your, true, false);
  data.candidates.forEach((c, i) => mkRow(T_MARKS[i], c, false, i === data.model_pick));
  $("t-note").innerHTML = data.match
    ? "和 AI 选得一样"
    : `AI 更倾向 ${T_MARKS[data.model_pick]}（${pct(data.candidates[data.model_pick].p_good)}）`;
}

function setBlocksMode(mode) {
  if (state.blocksMode === mode) return;
  state.blocksMode = mode;
  $("t-mode-auto").classList.toggle("active", mode === "auto");
  $("t-mode-manual").classList.toggle("active", mode === "manual");
  for (const id of ["t-left", "t-right", "t-rotate", "t-drop"]) {
    $(id).disabled = mode !== "manual";
  }
  $("t-cands-label").textContent = mode === "manual" ? "你的落点 / AI 的候选（★ 为 AI 首选）" : "候选落点 · AI 打分";
  $("t-s-int-label").textContent = mode === "manual" ? "与 AI 一致" : "护栏纠正";
  $("t-keys").hidden = mode !== "manual";
  $("t-s-int").textContent = "0";
  if (mode === "manual") {
    if (state.blocks) {
      state.blocks.playing = false;
      $("blocks-toggle").disabled = true;
    }
    if (state.blocks && state._blocksSnap && state._blocksSnap.alive) manualSpawn();
  } else {
    manualStop();
    if (state.blocks && state._blocksSnap && state._blocksSnap.alive) {
      drawBlocks(state._blocksSnap, null);
      state.blocks.playing = true;
      $("blocks-toggle").disabled = false;
      $("blocks-toggle").textContent = "暂停";
      blocksLoop(state.blocks.gen);
    }
  }
}

document.addEventListener("keydown", (e) => {
  if ($("view-blocks").hidden || state.blocksMode !== "manual" || !man.active) return;
  const k = KEYMAP[e.code];
  if (k === "left") manualTry(man.rot, man.col - 1);
  else if (k === "right") manualTry(man.rot, man.col + 1);
  else if (k === "up") manualRotate();
  else if (k === "down" || k === "space") manualLock();
  else return;
  e.preventDefault();
});

/* ---------- manual play: snake / bird / bricks ---------- */

const MANUAL_UI = {
  snake: { seg: "s", keys: "s-keys", intLabel: "s-int-label", pads: ["s-left", "s-up", "s-down", "s-right"] },
  bird: { seg: "f", keys: "f-keys", intLabel: "f-s-int-label", pads: ["f-up", "f-down"] },
  bricks: { seg: "b", keys: "b-keys", intLabel: "b-s-int-label", pads: ["b-left", "b-hold", "b-right"] },
};
const manual = { snake: false, bird: false, bricks: false };
const input = { snakeQueue: [], birdFlap: false, held: new Set() };
const OPPOSITE = { UP: "DOWN", DOWN: "UP", LEFT: "RIGHT", RIGHT: "LEFT" };

function setManual(game, on) {
  manual[game] = on;
  const ui = MANUAL_UI[game];
  $(`${ui.seg}-mode-auto`).classList.toggle("active", !on);
  $(`${ui.seg}-mode-manual`).classList.toggle("active", on);
  $(ui.keys).hidden = !on;
  $(ui.intLabel).textContent = on ? "与 AI 一致" : "护栏纠正";
  for (const id of ui.pads) $(id).disabled = !on;
  input.snakeQueue.length = 0;
  input.birdFlap = false;
  input.held.clear();
}

function snakeHeading(snap) {
  if (!snap || snap.body.length < 2) return "RIGHT";
  const dx = snap.body[0][0] - snap.body[1][0], dy = snap.body[0][1] - snap.body[1][1];
  return dx > 0 ? "RIGHT" : dx < 0 ? "LEFT" : dy > 0 ? "DOWN" : "UP";
}

// The action the player asked for since the last step; undefined lets the AI drive.
function manualAction(game) {
  if (!manual[game]) return undefined;
  if (game === "snake") {
    const heading = snakeHeading(state._snakeSnap);
    while (input.snakeQueue.length) {
      const want = input.snakeQueue.shift();
      if (want !== heading && want !== OPPOSITE[heading]) return want;
    }
    return heading;
  }
  if (game === "bird") {
    const flap = input.birdFlap;
    input.birdFlap = false;
    return flap ? "up" : "down";
  }
  const left = input.held.has("left"), right = input.held.has("right");
  return left && !right ? "left" : right && !left ? "right" : "hold";
}

const KEYMAP = {
  ArrowUp: "up", KeyW: "up", ArrowDown: "down", KeyS: "down",
  ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right",
  Space: "space", KeyJ: "fire",
};

function activeView() {
  return ["decide", "snake", "bird", "blocks", "bricks", "tank"].find((v) => !$("view-" + v).hidden);
}

function queueSnake(dir) {
  const q = input.snakeQueue;
  if (q[q.length - 1] !== dir && q.length < 3) q.push(dir);
  pressFx("s-" + dir.toLowerCase());
}

function birdFlap() {
  input.birdFlap = true;
  pressFx("f-up");
}

document.addEventListener("keydown", (e) => {
  const k = KEYMAP[e.code];
  if (!k || e.target.matches("textarea, input, select")) return;
  const view = activeView();
  if (view === "snake" && manual.snake && ["up", "down", "left", "right"].includes(k)) {
    queueSnake(k.toUpperCase());
  } else if (view === "bird" && manual.bird && (k === "space" || k === "up")) {
    if (!e.repeat) birdFlap();
  } else if (view === "bricks" && manual.bricks && (k === "left" || k === "right")) {
    input.held.add(k);
  } else {
    return;
  }
  e.preventDefault();
});
document.addEventListener("keyup", (e) => {
  const k = KEYMAP[e.code];
  if (k === "left" || k === "right") input.held.delete(k);
});
window.addEventListener("blur", () => input.held.clear());

for (const d of ["up", "down", "left", "right"]) {
  $("s-" + d).addEventListener("click", () => queueSnake(d.toUpperCase()));
}
$("f-up").addEventListener("pointerdown", birdFlap);
for (const side of ["left", "right"]) {
  const b = $("b-" + side);
  b.addEventListener("pointerdown", () => input.held.add(side));
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, () => input.held.delete(side));
}
$("b-hold").addEventListener("pointerdown", () => input.held.clear());
for (const game of ["snake", "bird", "bricks"]) {
  const seg = MANUAL_UI[game].seg;
  $(`${seg}-mode-auto`).onclick = () => setManual(game, false);
  $(`${seg}-mode-manual`).onclick = () => setManual(game, true);
}

/* ---------- shell ---------- */

function showError(id, msg) { const el = $(id); el.textContent = msg; el.hidden = false; }
function hide(id) { $(id).hidden = true; }

function switchView(name) {
  for (const v of ["decide", "snake", "bird", "blocks", "bricks", "tank"]) {
    $("view-" + v).hidden = name !== v;
    $("nav-" + v).classList.toggle("active", name === v);
  }
}

function effectiveTheme() {
  const t = document.documentElement.dataset.theme;
  if (t === "dark" || t === "light") return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  $("theme-toggle").textContent = effectiveTheme() === "dark" ? "☀" : "☾";
  if (state._snakeSnap) drawBoard(state._snakeSnap);
  if (state._birdSnap) drawBird(state._birdSnap, state._birdFrame);
  if (state._blocksSnap) drawBlocks(state._blocksSnap, state._blocksPlaced);
  if (state._bricksSnap) drawBricks(state._bricksSnap, state._bricksFrame);
}
$("theme-toggle").onclick = () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("laya-theme", next); } catch {}
};
try {
  const saved = localStorage.getItem("laya-theme");
  applyTheme(saved === "dark" || saved === "light" ? saved : null);
} catch { applyTheme(null); }

$("nav-decide").onclick = () => switchView("decide");
$("nav-snake").onclick = () => switchView("snake");
$("nav-bird").onclick = () => switchView("bird");
$("nav-blocks").onclick = () => switchView("blocks");
$("nav-bricks").onclick = () => switchView("bricks");
$("nav-tank").onclick = () => switchView("tank");
$("run").onclick = runPredict;
$("load-sample").onclick = loadSample;
$("snake-new").onclick = snakeNew;
$("snake-toggle").onclick = snakeToggle;
$("bird-new").onclick = birdNew;
$("bird-toggle").onclick = birdToggle;
$("blocks-new").onclick = blocksNew;
$("blocks-toggle").onclick = blocksToggle;
$("t-mode-auto").onclick = () => setBlocksMode("auto");
$("t-mode-manual").onclick = () => setBlocksMode("manual");
$("t-left").onclick = () => manualTry(man.rot, man.col - 1);
$("t-right").onclick = () => manualTry(man.rot, man.col + 1);
$("t-rotate").onclick = manualRotate;
$("t-drop").onclick = manualLock;
$("bricks-new").onclick = bricksNew;
$("bricks-toggle").onclick = bricksToggle;

initDirRows();
initBirdRows();
initBricksRows();
(async () => { await refreshInfo(); if (state.current) loadSample(); })();
setInterval(() => refreshInfo(true), 15000);
