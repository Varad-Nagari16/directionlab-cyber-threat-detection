import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Binary,
  BookOpen,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleDot,
  Clock3,
  Database,
  FileCheck2,
  FlaskConical,
  Gauge,
  GitBranch,
  Layers3,
  LockKeyhole,
  Menu,
  Network,
  Radar,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  X,
  Zap,
} from "lucide-react";

const navItems = [
  { label: "Framework", href: "#framework" },
  { label: "Method", href: "#method" },
  { label: "Roadmap", href: "#roadmap" },
  { label: "Evidence", href: "#evidence" },
];

const phases = [
  {
    number: "01",
    label: "Scope",
    title: "Freeze the research contract",
    copy: "Define what the detector can see, what it can never use, and which claims must wait for measured evidence.",
    state: "Active",
    accent: "cyan",
  },
  {
    number: "02",
    label: "Ingest",
    title: "Normalize one-way flow records",
    copy: "Convert the primary dataset into an auditable canonical schema without manufacturing a reverse flow.",
    state: "Next",
    accent: "lime",
  },
  {
    number: "03",
    label: "Model",
    title: "Compare static and causal encoders",
    copy: "Establish classical baselines, then test whether bounded temporal context adds signal without future leakage.",
    state: "Queued",
    accent: "violet",
  },
  {
    number: "04",
    label: "Validate",
    title: "Stress-test generalisation",
    copy: "Use chronological, host-controlled, and external-dataset splits to expose memorisation and optimistic bias.",
    state: "Queued",
    accent: "orange",
  },
];

const metrics = [
  { label: "Observed direction", value: "1", suffix: "way", icon: ArrowUpRight },
  { label: "Payloads inspected", value: "0", suffix: "bytes", icon: LockKeyhole },
  { label: "Reverse-flow features", value: "0", suffix: "allowed", icon: ArrowDownRight },
  { label: "Experimental results", value: "TBD", suffix: "until measured", icon: FlaskConical },
];

function scrollToId(id: string) {
  document.querySelector(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [openPhase, setOpenPhase] = useState(0);
  const [activeSection, setActiveSection] = useState("framework");

  useEffect(() => {
    const targets = navItems
      .map((item) => document.querySelector(item.href))
      .filter(Boolean) as Element[];
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActiveSection(visible.target.id);
      },
      { rootMargin: "-18% 0px -62% 0px", threshold: [0.1, 0.35, 0.7] },
    );
    targets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="site-shell">
      <div className="ambient-grid" aria-hidden="true" />
      <header className="topbar">
        <a className="brand" href="#top" aria-label="DirectionLab home">
          <span className="brand-mark"><Radar size={18} strokeWidth={2.4} /></span>
          <span>
            <strong>DirectionLab</strong>
            <small>research system / 01</small>
          </span>
        </a>
        <nav className={menuOpen ? "main-nav open" : "main-nav"} aria-label="Primary navigation">
          {navItems.map((item) => (
            <a
              key={item.href}
              className={activeSection === item.href.slice(1) ? "active" : ""}
              href={item.href}
              onClick={() => setMenuOpen(false)}
            >
              <span className="nav-index">0{navItems.indexOf(item) + 1}</span>{item.label}
            </a>
          ))}
          <a className="nav-cta" href="#roadmap" onClick={() => setMenuOpen(false)}>View roadmap <ArrowUpRight size={15} /></a>
        </nav>
        <button className="menu-toggle" onClick={() => setMenuOpen((value) => !value)} aria-label="Toggle navigation" aria-expanded={menuOpen}>
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </header>

      <main id="top">
        <section className="hero section-pad" id="framework">
          <div className="hero-copy">
            <div className="eyebrow"><span className="pulse-dot" /> research framework <span className="eyebrow-divider" /> pre-experiment phase</div>
            <h1>Detect threats.<br /><em>Keep direction.</em></h1>
            <p className="hero-lede">An AI framework for finding cyber threats in unidirectional IP traffic using only what a one-way monitor can actually observe.</p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => scrollToId("#method")}>Explore the method <ArrowUpRight size={17} /></button>
              <button className="text-button" onClick={() => scrollToId("#evidence")}>Read the evidence policy <ArrowDownRight size={17} /></button>
            </div>
            <div className="hero-footnote"><Check size={14} /> No experimental results claimed at this stage</div>
          </div>

          <div className="hero-visual" aria-label="Illustration of a one-way traffic observation pipeline">
            <div className="visual-header"><span>LIVE OBSERVATION MODEL</span><span className="live-indicator"><span /> read-only</span></div>
            <div className="radar-frame">
              <div className="radar-sweep" />
              <div className="radar-circle circle-one" />
              <div className="radar-circle circle-two" />
              <div className="radar-circle circle-three" />
              <div className="radar-cross cross-x" />
              <div className="radar-cross cross-y" />
              <div className="radar-origin"><CircleDot size={12} /></div>
              <div className="signal signal-a"><span className="signal-tag">FLOW_042</span><span className="signal-line" /><span className="signal-node" /></div>
              <div className="signal signal-b"><span className="signal-tag">FLOW_117</span><span className="signal-line" /><span className="signal-node" /></div>
              <div className="signal signal-c"><span className="signal-tag">FLOW_203</span><span className="signal-line" /><span className="signal-node" /></div>
              <div className="radar-label label-top">CURRENT FLOW</div>
              <div className="radar-label label-bottom">CAUSAL HISTORY / 60 SEC</div>
            </div>
            <div className="visual-footer"><span><span className="legend-dot cyan" /> metadata only</span><span><span className="legend-dot lime" /> direction locked</span><span><span className="legend-dot violet" /> temporal context</span></div>
          </div>
        </section>

        <section className="metric-strip section-pad" aria-label="Research constraints">
          {metrics.map((metric) => {
            const Icon = metric.icon;
            return <div className="metric" key={metric.label}><Icon size={16} /><div><strong>{metric.value}</strong><span>{metric.suffix}</span></div><p>{metric.label}</p></div>;
          })}
        </section>

        <section className="manifesto section-pad">
          <div className="section-kicker"><span>01</span><span>Why this matters</span></div>
          <div className="manifesto-grid">
            <h2>Most network research sees the whole conversation.<br /><em>This project starts with half of it.</em></h2>
            <div className="manifesto-copy"><p>One-way taps, privacy constraints, encrypted traffic, and constrained monitoring environments all create the same research pressure: make the strongest decision possible from the evidence that is actually present.</p><p>DirectionLab turns that constraint into a contract. Every feature, temporal window, split, and alert must respect the observed source-to-destination direction.</p></div>
          </div>
        </section>

        <section className="method section-pad" id="method">
          <div className="section-heading"><div><div className="section-kicker"><span>02</span><span>Method architecture</span></div><h2>Read-only by design.</h2></div><p>The detector is built as a chain of auditable decisions. Each layer earns its place by preserving causal validity.</p></div>
          <div className="architecture-card">
            <div className="architecture-top"><span>PIPELINE / v0.1</span><span className="architecture-status"><span className="pulse-dot" /> contract enforced</span></div>
            <div className="architecture-flow">
              <div className="arch-node input-node"><div className="node-icon"><Network size={20} /></div><span className="node-index">01 / INPUT</span><strong>Current<br />directional flow</strong><small>~64 observed features</small></div>
              <div className="flow-arrow"><span /></div>
              <div className="arch-node"><div className="node-icon"><Layers3 size={20} /></div><span className="node-index">02 / ENCODE</span><strong>Parallel<br />encoders</strong><small>Static MLP + causal TCN</small></div>
              <div className="flow-arrow"><span /></div>
              <div className="arch-node fusion-node"><div className="node-icon"><BrainCircuit size={20} /></div><span className="node-index">03 / FUSE</span><strong>Gated<br />fusion</strong><small>shared 64-dim representation</small></div>
              <div className="flow-arrow"><span /></div>
              <div className="arch-node"><div className="node-icon"><ShieldCheck size={20} /></div><span className="node-index">04 / DECIDE</span><strong>Binary +<br />multiclass</strong><small>calibrated probabilities</small></div>
              <div className="flow-arrow"><span /></div>
              <div className="arch-node output-node"><div className="node-icon"><Activity size={20} /></div><span className="node-index">05 / ALERT</span><strong>Evidence-<br />bearing JSON</strong><small>SOC-ready schema</small></div>
            </div>
            <div className="architecture-rules"><span><Check size={13} /> no reverse-flow features</span><span><Check size={13} /> no payload inspection</span><span><Check size={13} /> temporally causal</span><span><Check size={13} /> calibration included</span></div>
          </div>
        </section>

        <section className="features section-pad">
          <div className="section-heading"><div><div className="section-kicker"><span>03</span><span>What the system measures</span></div><h2>Signal without the payload.</h2></div><p>Feature families are intentionally close to the wire: observable, inspectable, and difficult to confuse with post-event knowledge.</p></div>
          <div className="feature-grid">
            <article className="feature-card feature-large"><div className="feature-card-top"><span className="feature-number">A</span><span className="feature-type">volume / rate</span></div><Gauge size={30} /><h3>Volumetric behaviour</h3><p>Duration, bytes, packets, and derived rates form the first view of traffic intensity. They are the baseline signal for burst, scan, and saturation patterns.</p><div className="mini-bars"><span style={{ height: "28%" }} /><span style={{ height: "42%" }} /><span style={{ height: "36%" }} /><span style={{ height: "68%" }} /><span style={{ height: "54%" }} /><span style={{ height: "88%" }} /><span style={{ height: "74%" }} /><span style={{ height: "100%" }} /></div></article>
            <article className="feature-card"><div className="feature-card-top"><span className="feature-number">B</span><span className="feature-type">temporal / causal</span></div><Clock3 size={26} /><h3>Bounded history</h3><p>Previous observations only. A causal TCN tests whether local context adds meaningful signal.</p><div className="timeline"><span /><span /><span /><span /><i /></div></article>
            <article className="feature-card"><div className="feature-card-top"><span className="feature-number">C</span><span className="feature-type">transport / port</span></div><Binary size={26} /><h3>Protocol metadata</h3><p>Ports, transport flags, and protocol context stay directional and payload-independent.</p><div className="code-lines"><span>tcp  →  443</span><span>syn   fin   ack</span><span>udp  →  53</span></div></article>
            <article className="feature-card"><div className="feature-card-top"><span className="feature-number">D</span><span className="feature-type">calibration / alert</span></div><FileCheck2 size={26} /><h3>Actionable uncertainty</h3><p>Scores become calibrated confidence, severity, and traceable evidence rather than an opaque class label.</p><div className="confidence-row"><span>confidence</span><strong>0.91</strong><div><i /></div></div></article>
          </div>
        </section>

        <section className="roadmap section-pad" id="roadmap">
          <div className="section-heading"><div><div className="section-kicker"><span>04</span><span>Build sequence</span></div><h2>From contract to evidence.</h2></div><p>Every milestone reduces uncertainty before the next layer is added. Open a phase to see its acceptance condition.</p></div>
          <div className="roadmap-layout"><div className="roadmap-line" />{phases.map((phase, index) => <button className={`phase-row ${openPhase === index ? "selected" : ""}`} key={phase.number} onClick={() => setOpenPhase(openPhase === index ? -1 : index)}><span className={`phase-marker ${phase.accent}`}>{openPhase === index ? <Check size={15} /> : phase.number}</span><span className="phase-main"><span className="phase-label">{phase.label} <i /> {phase.state}</span><strong>{phase.title}</strong><span className="phase-copy">{phase.copy}</span></span><ChevronDown className={openPhase === index ? "phase-chevron rotated" : "phase-chevron"} size={18} /></button>)}</div>
        </section>

        <section className="evidence section-pad" id="evidence">
          <div className="evidence-panel"><div className="evidence-header"><div className="section-kicker"><span>05</span><span>Evidence policy</span></div><span className="status-chip"><Sparkles size={13} /> no results yet</span></div><div className="evidence-content"><div><h2>Targets are not<br /><em>measurements.</em></h2><p>The project will publish measured values only after the dataset, preprocessing, split controls, and experiments are complete. Until then, every number is explicitly marked as a target, example, or TBD.</p><a href="#references" className="reference-link">See research references <ArrowUpRight size={15} /></a></div><div className="evidence-table"><div className="table-head"><span>measure</span><span>target</span><span>measured</span></div>{[["Average precision", "ranking quality", "TBD"], ["Macro-F1 / MCC", "imbalanced classes", "TBD"], ["Recall at fixed FPR", "operational trade-off", "TBD"], ["Detection delay", "campaign timelines", "TBD"]].map((row) => <div className="table-row" key={row[0]}><strong>{row[0]}</strong><span>{row[1]}</span><b>{row[2]}</b></div>)}</div></div></div>
        </section>

        <section className="closing section-pad"><div className="closing-mark"><Zap size={23} /></div><div><div className="eyebrow">the first milestone</div><h2>Build the data slice<br /><em>you can trust.</em></h2><p>Before training a model, prove that one row means one observed direction—and that no future or reverse-flow evidence slipped in.</p><button className="primary-button" onClick={() => scrollToId("#roadmap")}>Open the build sequence <ArrowUpRight size={17} /></button></div><div className="closing-terminal"><div><span className="terminal-dot red" /><span className="terminal-dot yellow" /><span className="terminal-dot green" /></div><pre><code><span className="code-muted">$ directionlab check</span>{"\n"}<span className="code-green">✓ schema contract loaded</span>{"\n"}<span className="code-green">✓ reverse-flow fields: 0</span>{"\n"}<span className="code-green">✓ payload fields: 0</span>{"\n"}<span className="code-cyan">→ next: ingest primary dataset</span></code></pre></div></section>
      </main>

      <footer className="footer section-pad" id="references"><div className="footer-brand"><span className="brand-mark"><Radar size={18} strokeWidth={2.4} /></span><div><strong>DirectionLab</strong><p>AI-based detection for unidirectional IP traffic.</p></div></div><div className="footer-links"><span>research system / 01</span><a href="https://nesg.ugr.es/nesg-ugr16/dataset_AuthorVersionFinal.pdf" target="_blank" rel="noreferrer"><BookOpen size={14} /> UGR'16 reference</a><a href="https://www.unb.ca/cic/datasets/ids-2017.html" target="_blank" rel="noreferrer"><Database size={14} /> CIC-IDS2017 reference</a></div><div className="footer-bottom"><span>Designed for reproducibility.</span><span>© 2026 DirectionLab / pre-experiment</span></div></footer>
    </div>
  );
}
