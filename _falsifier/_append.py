import json,os,sys
p='_falsifier/honesty_verdicts.json'
cur=json.load(open(p)) if os.path.exists(p) else []
new=json.load(sys.stdin)
seen={v['id'] for v in cur}
for v in new:
    if v['id'] in seen: raise SystemExit('dup id %d'%v['id'])
    cur.append(v)
cur.sort(key=lambda v:v['id'])
json.dump(cur,open(p,'w'),indent=1)
print('total verdicts:',len(cur))
