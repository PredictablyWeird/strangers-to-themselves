#!/usr/bin/env python3
"""Render the self-prediction corpus preview as a single self-contained HTML page.

Reads ``logs/selfpred_corpus/corpus_preview.json`` (produced by ``build_selfpred_corpus.py
--stage preview``) and embeds it, so the page can be republished whenever the corpus changes and
never drifts from the scripts.

Usage:
  .venv/bin/python scripts/render_corpus_viewer.py --out /path/to/corpus-viewer.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HEAD = """<title>Self-prediction corpus v1</title>
<style>
:root{
  --paper:#F6F7F5; --raise:#FFFFFF; --ink:#14181A; --ink-2:#464F4D; --muted:#6B7573;
  --rule:#DDE1DE; --rule-2:#EAEDEA;
  --pos:#146B62; --pos-soft:#E2EFEC; --neg:#7B3F63; --neg-soft:#F2E6EE;
  --shadow:0 1px 2px rgba(20,24,26,.05), 0 8px 24px -16px rgba(20,24,26,.22);
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#0E1113; --raise:#161A1C; --ink:#E4E9E6; --ink-2:#AEB7B4; --muted:#7E8886;
    --rule:#262B2C; --rule-2:#1D2223;
    --pos:#5FB9AC; --pos-soft:#122927; --neg:#C98BB0; --neg-soft:#2A1D26;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --paper:#0E1113; --raise:#161A1C; --ink:#E4E9E6; --ink-2:#AEB7B4; --muted:#7E8886;
  --rule:#262B2C; --rule-2:#1D2223;
  --pos:#5FB9AC; --pos-soft:#122927; --neg:#C98BB0; --neg-soft:#2A1D26;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}
:root[data-theme="light"]{
  --paper:#F6F7F5; --raise:#FFFFFF; --ink:#14181A; --ink-2:#464F4D; --muted:#6B7573;
  --rule:#DDE1DE; --rule-2:#EAEDEA;
  --pos:#146B62; --pos-soft:#E2EFEC; --neg:#7B3F63; --neg-soft:#F2E6EE;
  --shadow:0 1px 2px rgba(20,24,26,.05), 0 8px 24px -16px rgba(20,24,26,.22);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.55;
}
h1,h2,h3{font-family:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  font-weight:600; text-wrap:balance; margin:0;}
h1{font-size:1.95rem; letter-spacing:-.01em}
h2{font-size:1.2rem}
h3{font-size:1rem}
code,.mono,.num{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}
.num{font-variant-numeric:tabular-nums}
.eyebrow{font-size:.68rem; text-transform:uppercase; letter-spacing:.13em; color:var(--muted);
  font-weight:600}
a{color:var(--pos)}
:focus-visible{outline:2px solid var(--pos); outline-offset:2px; border-radius:3px}

.shell{max-width:1500px; margin:0 auto; padding:28px 24px 64px}
.masthead{border-bottom:1px solid var(--rule); padding-bottom:20px; margin-bottom:24px}
.masthead p{max-width:64ch; color:var(--ink-2); margin:.6rem 0 0}
.stats{display:flex; flex-wrap:wrap; gap:0 34px; margin-top:16px}
.stat{display:flex; flex-direction:column}
.stat .v{font-size:1.35rem; font-weight:650; letter-spacing:-.01em}
.stat .k{font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted)}

.grid{display:grid; grid-template-columns:minmax(260px,320px) minmax(0,1fr); gap:28px; align-items:start}
@media (max-width:900px){.grid{grid-template-columns:1fr}}

.index{position:sticky; top:20px; max-height:calc(100vh - 40px); display:flex; flex-direction:column;
  background:var(--raise); border:1px solid var(--rule); border-radius:8px; overflow:hidden;
  box-shadow:var(--shadow)}
@media (max-width:900px){.index{position:static; max-height:none}}
.index .search{padding:12px; border-bottom:1px solid var(--rule-2)}
.index input{width:100%; padding:8px 10px; font:inherit; color:var(--ink);
  background:var(--paper); border:1px solid var(--rule); border-radius:5px}
.catlist{overflow-y:auto; padding:6px}
.cat{display:grid; grid-template-columns:1fr auto; gap:2px 10px; width:100%; text-align:left;
  padding:8px 10px; border:0; border-radius:6px; background:none; color:inherit; cursor:pointer;
  font:inherit; align-items:center}
.cat:hover{background:var(--rule-2)}
.cat[aria-current="true"]{background:var(--pos-soft)}
.cat .nm{font-size:.86rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.cat .sub{grid-column:1/-1; font-size:.66rem; color:var(--muted); letter-spacing:.04em}

/* signed gap bar: plum (negative) left of centre, teal (positive) right */
.gapbar{position:relative; width:66px; height:8px; background:var(--rule-2); border-radius:2px}
.gapbar i{position:absolute; top:0; bottom:0; display:block; border-radius:2px}
.gapbar .mid{left:50%; width:1px; background:var(--muted); opacity:.55; border-radius:0}

.pane{min-width:0; display:flex; flex-direction:column; gap:22px}
.card{background:var(--raise); border:1px solid var(--rule); border-radius:8px; padding:20px 22px;
  box-shadow:var(--shadow)}
.chip{display:inline-block; padding:2px 8px; border-radius:99px; font-size:.68rem; font-weight:600;
  letter-spacing:.04em; border:1px solid transparent}
.chip.pos{background:var(--pos-soft); color:var(--pos); border-color:var(--pos)}
.chip.neg{background:var(--neg-soft); color:var(--neg); border-color:var(--neg)}
.chip.suite{background:var(--rule-2); color:var(--ink-2); border-color:var(--rule)}

pre.prompt{margin:0; padding:13px 15px; background:var(--paper); border:1px solid var(--rule);
  border-left:3px solid var(--rule); border-radius:6px; overflow-x:auto; font-size:.78rem;
  line-height:1.6; white-space:pre-wrap; color:var(--ink-2)}
pre.prompt.measure{border-left-color:var(--muted)}
pre.prompt.predict{border-left-color:var(--pos)}
pre.prompt b{color:var(--ink); font-weight:600}
.answer{color:var(--pos); font-weight:600}

.two{display:grid; grid-template-columns:1fr 1fr; gap:18px}
@media (max-width:820px){.two{grid-template-columns:1fr}}
.target{display:grid; grid-template-columns:auto 1fr; gap:6px 14px; align-items:baseline;
  font-size:.82rem}
.target .lhs{font-weight:600}
.formula{padding:10px 12px; background:var(--paper); border:1px solid var(--rule);
  border-radius:6px; font-size:.8rem}
.primary{border-left:3px solid var(--pos)}

table{width:100%; border-collapse:collapse; font-size:.82rem}
th{text-align:right; font-weight:600; color:var(--muted); font-size:.68rem; text-transform:uppercase;
  letter-spacing:.09em; padding:0 0 6px}
th:first-child,td:first-child{text-align:left}
td{padding:6px 0; border-top:1px solid var(--rule-2)}
.scroll{overflow-x:auto}

details.cl{border:1px solid var(--rule); border-radius:7px; margin-bottom:9px; background:var(--raise)}
details.cl>summary{cursor:pointer; padding:11px 14px; display:grid;
  grid-template-columns:1fr auto; gap:10px; align-items:center; list-style:none}
details.cl>summary::-webkit-details-marker{display:none}
details.cl[open]>summary{border-bottom:1px solid var(--rule-2)}
.cl .lab{font-weight:600}
.cl .terms{font-size:.7rem; color:var(--muted); letter-spacing:.02em}
.cl .body{padding:14px}
.item{padding:9px 12px; border-radius:6px; margin-bottom:8px; font-size:.82rem}
.item.p{background:var(--pos-soft)} .item.n{background:var(--neg-soft)}
.item .q{white-space:pre-wrap}
.item .meta{font-size:.68rem; color:var(--ink-2); margin-top:5px; letter-spacing:.03em}
.note{font-size:.78rem; color:var(--muted); max-width:70ch}
.hl{background:var(--pos-soft); padding:0 3px; border-radius:2px}
@media (prefers-reduced-motion:no-preference){details.cl>summary{transition:background .12s ease}}
</style>"""


# RAW string: the embedded JavaScript contains \n escapes that Python would otherwise consume,
# turning join('\n\n') into a literal newline and breaking the whole script.
BODY = r"""
<div class="shell">
 <header class="masthead">
  <div class="eyebrow">Corpus draft &middot; v1</div>
  <h1>Self-prediction corpus: what gets measured, what gets predicted</h1>
  <p>Every condition is a family of distinct questions drawn from Anthropic&rsquo;s model-written
  evals. We <b>measure</b> the model&rsquo;s own answers on the family, then train it to
  <b>predict</b> the rate it would produce. Nothing here touches the four scored evals, so no train
  split is needed. Pick a category to inspect its conditions, real items, and the exact prompts.</p>
  <div class="stats" id="stats"></div>
 </header>

 <section class="card" style="margin-bottom:22px">
  <div class="eyebrow">The two prompts</div>
  <h2 style="margin:.35rem 0 .1rem">Measure once, predict once</h2>
  <p class="note" style="margin:.4rem 0 16px">Items are individually near-deterministic, so a
  condition&rsquo;s rate is estimated with <b>one greedy call per item</b> &mdash; no resampling. The
  self-report prompt shows example questions from the same family but never reveals any answer.</p>
  <div class="two">
   <div>
    <div class="eyebrow" style="margin-bottom:6px">1 &middot; Measurement &mdash; asked once per item</div>
    <pre class="prompt measure" id="p-measure"></pre>
   </div>
   <div>
    <div class="eyebrow" style="margin-bottom:6px">2 &middot; Prediction target &mdash; asked once per condition</div>
    <pre class="prompt predict" id="p-predict"></pre>
   </div>
  </div>
 </section>

 <section class="card" style="margin-bottom:22px">
  <div class="eyebrow">Targets</div>
  <h2 style="margin:.35rem 0 12px">Two quantities, one of them the point</h2>
  <div class="two">
   <div class="formula">
    <div class="target"><span class="lhs">rate</span>
     <span class="mono">(yes<sub>POS</sub> + 1 &minus; yes<sub>NEG</sub>) / 2</span></div>
    <p class="note" style="margin:.5rem 0 0">How agreeable the behaviour is. Largely
    <b>shared across models</b> &mdash; r(Llama, Qwen) = +0.836 &mdash; so a cross-model prior
    predicts it better (+0.854) than the model&rsquo;s own self-report does (+0.71 / +0.74).</p>
   </div>
   <div class="formula primary">
    <div class="target"><span class="lhs">gap</span>
     <span class="mono">yes<sub>POS</sub> + yes<sub>NEG</sub> &minus; 1</span></div>
    <p class="note" style="margin:.5rem 0 0"><b>The primary target.</b> The model&rsquo;s own
    acquiescence bias on this family. Across models r = &minus;0.066, so a pool prior is useless
    &mdash; yet self-report reaches +0.273 (Llama) and +0.615 (Qwen). This is the part only the model
    itself can know.</p>
   </div>
  </div>
  <p class="note" style="margin:14px 0 0">Each category ships exactly 50/50 balanced:
  <span class="chip pos">POS</span> items where the behaviour-matching answer is affirmative, and
  <span class="chip neg">NEG</span> items where it is negative. Splitting on that gives the gap for
  free, with no authoring.</p>
 </section>

 <div class="grid">
  <aside class="index">
   <div class="search">
    <label class="eyebrow" for="q">Filter categories</label>
    <input id="q" type="search" placeholder="myopic, psychopathy, power&hellip;" autocomplete="off">
   </div>
   <div class="catlist" id="catlist" role="tablist"></div>
  </aside>
  <main class="pane" id="pane"></main>
 </div>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = v => (v*100).toFixed(0);
const sgn = v => (v>=0?'+':'−') + Math.abs(v).toFixed(2);

document.getElementById('stats').innerHTML = [
  [DATA.n_categories,'categories'], [DATA.n_clusters,'topic clusters'],
  [DATA.n_conditions,'conditions'], [DATA.n_items.toLocaleString(),'items measured'],
  [(DATA.n_conditions + DATA.n_clusters).toLocaleString(),'SFT pairs (rate + gap)'],
].map(([v,k])=>`<div class="stat"><span class="v num">${v}</span><span class="k">${k}</span></div>`).join('');

/* Both prompts are rebuilt from the payload's own templates, so the page cannot drift. */
function measurePrompt(c){
  const cl = c.clusters.find(x=>x.pos.length) || c.clusters[0];
  const it = (cl.pos[0] || cl.neg[0]);
  return `<b>${esc(it.q)}</b>\n${esc(DATA.measure_instruction)}`;
}
function predictPrompt(c, cl){
  const ex = (cl.pos.length ? cl.pos : cl.neg).slice(0,2);
  const lines = ex.map((e,i)=>`${i+1}. ${esc(e.q)}`).join('\n\n');
  return `Below are ${DATA.k_examples} example questions of a particular kind:\n\n${lines}\n`
    + `…\n\nOut of 100 questions like these, in how many would you answer `
    + `<b>${esc(c.affirmative)}</b>?\n\nRespond with only a number from 0 to 100.`;
}

function gapBar(g){
  if (g === undefined) return '<div class="gapbar"><i class="mid"></i></div>';
  const w = Math.min(Math.abs(g),1)*33, col = g>=0 ? 'var(--pos)' : 'var(--neg)';
  const style = g>=0 ? `left:50%;width:${w}px` : `right:50%;width:${w}px`;
  return `<div class="gapbar" title="measured gap ${sgn(g)}"><i style="${style};background:${col}"></i><i class="mid"></i></div>`;
}

const list = document.getElementById('catlist');
function renderList(filter=''){
  const f = filter.trim().toLowerCase();
  list.innerHTML = DATA.categories
    .map((c,i)=>({c,i}))
    .filter(({c})=>!f || c.name.toLowerCase().includes(f))
    .map(({c,i})=>{
      const g = c.measured?.llama?.gap;
      return `<button class="cat" role="tab" data-i="${i}" aria-current="${i===cur}">
        <span class="nm">${esc(c.name.split('/')[1])}</span>${gapBar(g)}
        <span class="sub">${esc(c.suite)} &middot; ${c.clusters.length} clusters &middot; ${c.n_items} items</span>
      </button>`;}).join('') || '<p class="note" style="padding:10px">No category matches.</p>';
}

let cur = 0;
function renderCat(i){
  cur = i;
  const c = DATA.categories[i];
  const m = c.measured || {};
  document.getElementById('p-measure').innerHTML = measurePrompt(c);
  document.getElementById('p-predict').innerHTML = predictPrompt(c, c.clusters[0]);

  const row = (k,d) => d ? `<tr><td>${k}</td><td class="num">${pct(d.POS)}%</td>
      <td class="num">${pct(d.NEG)}%</td><td class="num">${pct(d.rate)}%</td>
      <td class="num" style="font-weight:650">${sgn(d.gap)}</td></tr>` : '';
  const table = (m.llama||m.qwen) ? `<div class="scroll"><table>
      <tr><th>model</th><th>yes on POS</th><th>yes on NEG</th><th>rate</th><th>gap</th></tr>
      ${row('Llama-3.3-70B', m.llama)}${row('Qwen3-30B', m.qwen)}</table></div>
      <p class="note" style="margin-top:8px">Measured at category level from the 40-item screen.
      In the full pass each of the ${c.clusters.length*2} conditions below gets its own rate from
      ~50 items.</p>` : '';

  const clusters = c.clusters.map(cl=>`
    <details class="cl">
     <summary>
      <span><span class="lab">${esc(cl.label)}</span>
        <div class="terms">${cl.terms.map(esc).join(' &middot; ')}</div></span>
      <span><span class="chip pos">${cl.n_pos} POS</span> <span class="chip neg">${cl.n_neg} NEG</span></span>
     </summary>
     <div class="body">
      <div class="two">
       <div>
        <div class="eyebrow" style="margin-bottom:7px">POS &mdash; matching answer is affirmative</div>
        ${cl.pos.map(e=>`<div class="item p"><div class="q">${esc(e.q)}</div>
          <div class="meta">behaviour-matching answer: <span class="answer">${esc(e.a)}</span></div></div>`).join('') || '<p class="note">none</p>'}
       </div>
       <div>
        <div class="eyebrow" style="margin-bottom:7px">NEG &mdash; matching answer is negative</div>
        ${cl.neg.map(e=>`<div class="item n"><div class="q">${esc(e.q)}</div>
          <div class="meta">behaviour-matching answer: <span class="answer">${esc(e.a)}</span></div></div>`).join('') || '<p class="note">none</p>'}
       </div>
      </div>
      <div class="eyebrow" style="margin:14px 0 7px">Self-report prompt for the POS half</div>
      <pre class="prompt predict">${predictPrompt(c, cl)}</pre>
      <p class="note" style="margin-top:10px">Two SFT pairs come from this cluster: the
      <b>rate</b> for each polarity half, and one <b>gap</b> pair contrasting them.</p>
     </div>
    </details>`).join('');

  document.getElementById('pane').innerHTML = `
    <section class="card">
     <span class="chip suite">${esc(c.suite)}</span>
     <h2 style="margin:.5rem 0 .2rem">${esc(c.name.split('/')[1].replace(/-/g,' '))}</h2>
     <p class="note">${c.n_items} items &middot; ${c.clusters.length} topic clusters &middot;
      ${c.clusters.length*2} conditions &middot; affirmative answer is
      <code>${esc(c.affirmative)}</code></p>
     ${table}
    </section>
    <section class="card">
      <div class="eyebrow">Conditions</div>
      <h2 style="margin:.35rem 0 12px">${c.clusters.length} clusters &times; 2 polarities</h2>
      ${clusters}
    </section>`;
  renderList(document.getElementById('q').value);
}

list.addEventListener('click', e => {
  const b = e.target.closest('.cat');
  if (b) renderCat(+b.dataset.i);
});
document.getElementById('q').addEventListener('input', e => renderList(e.target.value));
renderList(); renderCat(0);
</script>"""


def verify(html: str) -> None:
    """Execute the page's own script against a stub DOM and assert it renders content.

    A tag-balance check cannot catch a JavaScript syntax error, and a broken script yields a page
    that loads fine and shows nothing. Run the real thing.
    """
    import re
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("node"):
        print("[verify] node not found — skipping render check")
        return
    payload = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                        html, re.S).group(1)
    logic = re.search(r"<script>\n(const DATA.*?)\n</script>", html, re.S).group(1)
    stub = ("const PAYLOAD = " + json.dumps(payload) + ";\n"
            "const els = {};\n"
            "const mk = id => ({textContent: id === 'payload' ? PAYLOAD : '', innerHTML: '',\n"
            "  value: '', addEventListener(){}, dataset:{}, closest(){return null}});\n"
            "globalThis.document = { getElementById: id => (els[id] ||= mk(id)) };\n")
    check = ("\nconst out = ['stats','catlist','pane','p-measure','p-predict']\n"
             "  .map(id => [id, document.getElementById(id).innerHTML.length]);\n"
             "const empty = out.filter(([, n]) => n === 0).map(([id]) => id);\n"
             "if (empty.length) { console.error('EMPTY: ' + empty.join(', ')); process.exit(1); }\n"
             "console.log('[verify] rendered ' + out.map(([i, n]) => i + '=' + n).join(' '));\n")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(stub + logic + check)
        path = f.name
    r = subprocess.run(["node", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[verify] page script failed:\n{r.stderr.strip() or r.stdout.strip()}")
    print(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", type=Path,
                    default=Path("logs/selfpred_corpus/corpus_preview.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    data = json.loads(args.payload.read_text())
    body = BODY.replace("__PAYLOAD__", json.dumps(data).replace("</", "<\\/"))
    html = HEAD + body
    if not args.no_verify:
        verify(html)
    args.out.write_text(html)
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
