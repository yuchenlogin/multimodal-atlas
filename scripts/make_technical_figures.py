#!/usr/bin/env python3
"""Original mechanism diagrams and plots from checked evidence / explicit teaching probes."""
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'/'technical';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfbf9','axes.facecolor':'#fbfbf9','savefig.facecolor':'#fbfbf9','axes.titleweight':'bold'})
BLUE='#244b8a';ORANGE='#b45c2b';GREEN='#2b7568'
# Node coordinates and typed edges make the forward, supervision, and feedback paths explicit.
SPECS={
'00':('Attention is a shaped computation',[('X: [B,T,d]',.07,.5),('Q: [B,h,T,dh]',.30,.79),('K: [B,h,T,dh]',.30,.5),('V: [B,h,T,dh]',.30,.21),('QK^T / sqrt(dh)\n[B,h,T,T] + mask',.58,.7),('softmax(S) @ V\n[B,h,T,dh]',.80,.5),('concat + output W\nresidual + MLP',.80,.17)],[(0,1,'Wq'),(0,2,'Wk'),(0,3,'Wv'),(1,4,''),(2,4,''),(4,5,'row softmax'),(3,5,''),(5,6,'')]),
'01':('Contrastive readout and generation use different visual interfaces',[
 ('image patches\n[B,Nv,dv]',.09,.78),
 ('vision encoder\nF: [B,Nv,dv]',.34,.78),
 ('pool + normalize\nimage: [B,D]',.59,.78),
 ('image-text logits\n[B,B] -> CLIP loss',.89,.78),
 ('text encoder + pool\ntext: [B,D]',.89,.23),
 ('queries / projector\n[B,Nq,d]',.34,.23),
 ('causal LLM\nanswer-only CE',.59,.23)],
 [(0,1,''),(1,2,''),(2,3,''),(4,3,'compare'),(1,5,'patch features'),(5,6,'prefix')]),
'02':('Residual coding separates rate from temporal order',[('waveform\n[B, samples]',.09,.65),('causal encoder\n[B,Tc,D]',.31,.65),('Q1 coarse code\nresidual r1',.58,.78),('Q2 ... Qq\nresidual rq',.90,.78),('code indices\n[B,Tc,Q]',.90,.28),('causal decoder\nwaveform estimate',.31,.28)],[(0,1,''),(1,2,'r0'),(2,3,'subtract'),(2,4,'index 1'),(3,4,'indices 2..Q'),(4,5,'sum embeddings')]),
'03':('A temporal model cannot recover an unsampled event',[('video + timestamps\ncontinuous scene',.10,.66),('sampling policy\nuniform / adaptive',.34,.66),('spatial encoder\n[F,Nv,D]',.59,.66),('temporal positions\n+ connector',.83,.66),('question\n+ token budget',.34,.22),('LLM answer\n+ temporal evidence',.83,.22)],[(0,1,'capture'),(1,2,'frames'),(2,3,'tokens'),(4,1,'allocate'),(3,5,'condition'),(4,5,'query')]),
'04':('Reasoning and speech run on different clocks',[('audio / video\nwith timestamps',.10,.72),('blockwise encoders\nshared time base',.34,.72),('Thinker\nsemantic state + text',.59,.72),('Talker\nspeech-code stream',.83,.72),('cancel / barge-in\nstate + buffers',.34,.22),('chunked DiT + vocoder\nwaveform packets',.83,.22)],[(0,1,''),(1,2,''),(2,3,'hidden + text'),(3,5,'codes'),(4,3,'interrupt'),(4,5,'invalidate')]),
'05':('Training sees noisy examples; sampling solves a trajectory',[('clean latent x0',.10,.75),('noise + schedule\nxt = a*x0 + b*eps',.36,.75),('DiT(xt,t,condition)\npredict eps or v',.65,.75),('regression loss\nknown eps / v target',.89,.75),('Gaussian xT',.10,.23),('reverse update\nK denoiser evaluations',.48,.23),('VAE decoder\npixels',.85,.23)],[(0,1,''),(1,2,''),(2,3,''),(1,3,'known target'),(4,5,''),(2,5,'same denoiser'),(5,6,'x0 estimate')]),
'06':('An action chunk is embedded in a physical feedback loop',[('camera + proprio\nobservation ot',.10,.7),('policy(observation, goal)\nchunk [H,da]',.38,.7),('denormalize\nframe + rate + limits',.69,.7),('servo + plant\nreal next state',.85,.25),('next observation\nreplan after k actions',.38,.25),('goal / task\nreference frame',.10,.25)],[(0,1,''),(5,1,''),(1,2,''),(2,3,'commands'),(3,4,'feedback'),(4,0,'sense')]),
'07':('Tool success is a state transition, not fluent text',[('snapshot + goal',.08,.72),('policy proposes\ntyped tool arguments',.34,.72),('contract checks\npermissions / schema',.61,.72),('tool executes\nreceipt / error',.85,.72),('verify postcondition\nagainst fresh state',.62,.23),('recover / checkpoint\nsafe retry only',.28,.23)],[(0,1,''),(1,2,''),(2,3,''),(3,4,'observe'),(4,0,'next step'),(4,5,'failed'),(5,0,'refresh')]),
'08':('Prefill and decode stress different resources',[('request + scheduler\nB sequences, T prefix',.09,.72),('prefill\nlarge matmuls',.38,.72),('decode\nsmall per-step matmuls',.72,.72),('HBM: weights + KV\nbytes moved',.72,.22),('GPU arithmetic\nFLOPs available',.38,.22),('end-to-end metrics\nTTFT / TPOT / p95',.09,.22)],[(0,1,''),(1,2,'cache ready'),(3,2,'read'),(4,1,'compute'),(1,5,'timing'),(2,5,'timing')]),
'09':('DPO compares policy and reference log probabilities',[('image + prompt\nchosen / rejected',.10,.75),('policy log probs\nanswer tokens only',.39,.78),('frozen reference\nlog probs',.39,.25),('difference of\nlog-ratios Delta',.65,.65),('softplus(-beta*Delta)\npolicy gradient',.93,.65),('image-swap control\nvisual dependence',.68,.18)],[(0,1,''),(0,2,''),(1,3,''),(2,3,''),(3,4,''),(0,5,'counterfactual')]),
'10':('A paired estimate needs item-level outcomes',[('frozen items\ncluster IDs + prompts',.10,.7),('system A\nfixed decoding',.38,.82),('system B\nfixed decoding',.38,.4),('same scoring rule\npaired differences',.67,.7),('cluster-aware resample\nuncertainty',.86,.25),('claim scope\nmetric + population',.38,.13)],[(0,1,''),(0,2,''),(1,3,''),(2,3,''),(3,4,''),(4,5,'interval')]),
'11':('Split by provenance before selecting training examples',[('raw records\nsource + license',.10,.74),('canonicalize\nhash / near duplicates',.38,.74),('family / source split\nfreeze held-out IDs',.71,.74),('train-only filtering\nmixture + budget',.71,.23),('training run\nfixed recipe',.38,.23),('locked evaluation\nno threshold tuning',.10,.23)],[(0,1,''),(1,2,''),(2,3,'train'),(3,4,''),(4,5,''),(2,5,'held-out')]),
'12':('Action-conditioned planning needs intervention checks',[('observation xt',.08,.74),('encoder zt\ntarget space',.33,.74),('dynamics(zt, action)\npredicted future',.62,.74),('goal cost\nsearch action sequence',.92,.74),('environment step\nactual action outcome',.65,.23),('held-out interventions\ncheck calibration',.30,.23)],[(0,1,''),(1,2,''),(2,3,''),(3,4,'first action'),(4,5,'measured'),(5,2,'test, not fit')]),
}
def save(fig,name):
 fig.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close(fig)
def mechanism(topic, spec):
 title, nodes, edges = spec
 fig, ax = plt.subplots(figsize=(15, 6.4))
 ax.set(xlim=(-.06, 1.06), ylim=(-.03, 1.03)); ax.axis('off')
 ax.set_title(title, loc='left', pad=18, fontsize=18)
 boxes = []
 for label, x, y in nodes:
  box = FancyBboxPatch((x-.095, y-.10), .19, .20,
      boxstyle='round,pad=.005', facecolor='#edf2fa', edgecolor=BLUE,
      linewidth=1.4, zorder=3)
  ax.add_patch(box); boxes.append(box)
  label = '\n'.join('\n'.join(textwrap.wrap(line, width=18, break_long_words=False)) for line in label.split('\n'))
  ax.text(x, y, label, ha='center', va='center', fontsize=14,
      color='#1a2233', zorder=4)
 for a, b, label in edges:
  _, x, y = nodes[a]; _, xx, yy = nodes[b]
  # Matplotlib clips the centre-to-centre segment at both actual node boundaries.
  # A fixed radial offset reverses arrows when nodes are closer than 2*offset.
  ax.add_patch(FancyArrowPatch((x,y), (xx,yy), patchA=boxes[a], patchB=boxes[b],
      shrinkA=4, shrinkB=4, arrowstyle='-|>', mutation_scale=14,
      color=ORANGE, linewidth=1.6, zorder=2,
      connectionstyle='arc3,rad=-0.45' if topic == '05' and (a,b) == (1,3) else 'arc3,rad=0'))
  if label and not (topic == '08' and label == 'timing'):
   lx, ly = ((.64,.98) if topic == '05' and (a,b) == (1,3) else ((x+xx)/2, (y+yy)/2+.035))
   ax.text(lx, ly, label, color=ORANGE, ha='center',
       fontsize=10, zorder=5, bbox={'facecolor':'#fbfbf9','edgecolor':'none','pad':1})
 fig.text(.02, .015, 'MECHANISM SCHEMATIC | original explanatory drawing; boxes are interfaces, not measured performance', fontsize=9, color='#556070')
 save(fig, f'{topic}-method.png')
def evidence(row):
 fig,ax=plt.subplots(figsize=(10,5));series=row['values'];labels=row['labels'];x=np.arange(len(labels));w=.75/len(series)
 for j,(values,label) in enumerate(zip(series,row['series'])):
  bars=ax.bar(x+(j-(len(series)-1)/2)*w,values,w,label=label,color=[BLUE,ORANGE,GREEN][j%3])
  ax.bar_label(bars,fmt='%g',padding=3,fontsize=10)
 ax.set_xticks(x,labels,rotation=12 if len(labels)>3 else 0);ax.set_ylabel(row['metric']);ax.set_ylim(0,max(map(max,series))*1.30)
 ax.set_title('Author-reported results | '+row['title'].split(':')[0],loc='left',fontsize=12,wrap=True);ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.02))
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.text(.02,.015,f"SOURCE {row['arxiv']} | {row['locator']}\nHistorical paper snapshot; see chapter caption and sources/evidence.json for protocol and caveats.",fontsize=8,color='#556070')
 fig.tight_layout(rect=(0,.08,1,1));save(fig,f"{row['topic']}-evidence.png")
def probe(topic,p):
 fig,ax=plt.subplots(figsize=(10,4.8));x=np.array(p['x'])
 if topic=='10':
  ax.hist(p['series']['Paired accuracy difference'],bins=35,color=BLUE,alpha=.85);ax.axvline(0,color=ORANGE,ls='--');ax.set(xlabel='B minus A accuracy',ylabel='Bootstrap count');ax.set_title(f"Constructed paired outcomes | estimate {p['estimate']:.3f}; 95% interval [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}]",loc='left',fontsize=11)
 elif topic=='12':
  bars=ax.bar(x,p['series']['Estimated effect'],color=[ORANGE,BLUE,GREEN]);ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_xticks(x,p['labels']);ax.set_ylabel(p['ylabel']);ax.set_title(p['kind'],loc='left')
 else:
  for j,(label,ys) in enumerate(p['series'].items()):ax.plot(x,ys,label=label,color=[BLUE,ORANGE,GREEN][j%3],lw=2)
  ax.set(xlabel=p['xlabel'],ylabel=p['ylabel']);ax.set_title(p['kind'],loc='left');ax.legend(frameon=False)
  if topic=='02':ax.set_yscale('log')
 ax.grid(alpha=.17);fig.text(.02,.01,'TEACHING PROBE | seeded CPU calculation / constructed system; not a foundation-model reproduction',fontsize=8,color='#556070');fig.tight_layout(rect=(0,.04,1,1));save(fig,f'{topic}-probe.png')
def cluster_figure():
 data = np.genfromtxt(ROOT/'sources/raw/check-10-bootstrap-draws.csv', delimiter=',', names=True)
 fig, ax = plt.subplots(figsize=(10,4.8))
 bins = np.linspace(-.35,.4,45)
 for column, label, color in [('item_bootstrap_difference','item resampling (ignores clusters)',BLUE), ('cluster_bootstrap_difference','cluster resampling (20 independent units)',ORANGE)]:
  ax.hist(data[column]*100, bins=bins*100, histtype='step', linewidth=2, label=label, color=color)
 ax.axvline(0, color='#555', ls=':'); ax.axvline(5, color=GREEN, ls='--', label='observed difference: 5 pp')
 ax.set(xlabel='B minus A accuracy (percentage points)', ylabel='Bootstrap count', title='Same 200 records; different independence assumptions')
 ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.15)
 fig.text(.02,.01,'CONSTRUCTED CLUSTERED DATA | 20 clusters x 10 identical outcomes; 4000 resamples per scheme',fontsize=8,color='#556070')
 fig.tight_layout(rect=(0,.04,1,1)); save(fig,'10-cluster.png')

def planning_figure():
 data = json.loads((ROOT/'sources/planning-probe.json').read_text())
 fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
 time = np.arange(81)*data['config']['dt']
 for name, color in [('PD', GREEN), ('fitted MPC', BLUE)]:
  axes[0].plot(time, np.array(data['first_trial_trajectories'][name])[:,0], label=name, color=color)
 axes[0].axhline(-1, color='#555', ls=':', label='goal')
 axes[0].set(title='Paired trial 0: working controls', xlabel='Seconds', ylabel='Position'); axes[0].legend(frameon=False)
 axes[1].plot(time, np.array(data['first_trial_trajectories']['shuffled-action MPC'])[:,0], color=ORANGE)
 axes[1].axhspan(-2, 2, alpha=.1, color=GREEN); axes[1].axhline(-1, color='#555', ls=':')
 axes[1].set(title='Same planner, shuffled actions', xlabel='Seconds', ylabel='Position (full range)')
 names = list(data['summary']); values = [data['summary'][n]['successes'] for n in names]
 bars = axes[2].bar(np.arange(3), values, color=[GREEN, BLUE, ORANGE]); axes[2].bar_label(bars, labels=[f'{v}/24' for v in values], padding=4)
 axes[2].set(xticks=np.arange(3), xticklabels=['PD','fitted\nMPC','shuffled\nMPC'], ylim=(0,29), ylabel='Successful episodes', title='Fixed 24-episode teaching suite')
 for ax in axes: ax.grid(alpha=.15); ax.set_axisbelow(True)
 fig.text(.01,.01,'EXECUTED CPU TOY | known linear simulator; least-squares dynamics, not a visual/robot model',fontsize=9,color='#556070')
 fig.tight_layout(rect=(0,.06,1,1)); save(fig,'12-planning.png')

def write_manifest():
 import hashlib
 rows = []
 for path in sorted(OUT.glob('*.png')):
  topic, kind = path.stem.split('-',1)
  data_file = {'method': None, 'probe': 'sources/probes.json', 'evidence': 'sources/evidence.json',
               'cluster': 'sources/mechanism-checks.json', 'planning': 'sources/planning-probe.json'}[kind]
  rows.append({'figure': path.relative_to(ROOT).as_posix(), 'topic': topic, 'kind': kind,
               'data': data_file, 'generator': 'scripts/make_technical_figures.py',
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
 (ROOT/'sources/figure-manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')

if __name__=='__main__':
 for topic,spec in SPECS.items():mechanism(topic,spec)
 for row in json.loads((ROOT/'sources/evidence.json').read_text(encoding='utf8')):evidence(row)
 for topic,p in json.loads((ROOT/'sources/probes.json').read_text(encoding='utf8'))['probes'].items():probe(topic,p)
 planning_figure()
 cluster_figure()
 write_manifest()
 print('Built',len(list(OUT.glob('*.png'))),'figures in',OUT)
