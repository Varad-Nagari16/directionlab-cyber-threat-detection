"""Plan host-held-out folds from prepared NetFlow v3 partitions."""
from __future__ import annotations
import argparse, json
from itertools import permutations
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root', default='data/processed/nfv3_directional'); parser.add_argument('--output', default='reports/nfv3_host_folds.json'); args=parser.parse_args()
    root=Path(args.data_root); frames=[]
    for name in ('train','calibration','test'):
        frame=pd.read_csv(root/f'{name}.csv', usecols=['source_host_id','label_binary'], low_memory=False); frame['partition']=name; frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    stats=[]
    for host,group in frame.groupby('source_host_id'):
        attacks=int(group['label_binary'].sum()); stats.append({'host_id':str(host),'rows':int(len(group)),'attacks':attacks,'source_partition':sorted(group['partition'].unique().tolist())})
    attack_hosts=sorted(item['host_id'] for item in stats if item['attacks']>0)
    folds=[]
    for calibration,test in permutations(attack_hosts,2):
        train=[host for host in attack_hosts if host not in {calibration,test}]
        folds.append({'train_attack_hosts':train,'calibration_attack_host':calibration,'test_attack_host':test})
    result={'host_count':len(stats),'attack_host_count':len(attack_hosts),'hosts':stats,'fold_count':len(folds),'folds':folds,'note':'Fold metrics must be computed with fixed thresholds selected on calibration hosts; do not tune on test hosts.'}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'host_count':len(stats),'attack_host_count':len(attack_hosts),'fold_count':len(folds)},indent=2))
if __name__=='__main__': main()
