import json,sys
lo,hi=int(sys.argv[1]),int(sys.argv[2])
for i,line in enumerate(open('_falsifier/honesty_blind.jsonl'),1):
    if lo<=i<=hi:
        d=json.loads(line)
        print("### id %d"%d['id'])
        print("PROMPT: %s"%d['prompt'])
        print("A: %s"%d['A'])
        print("B: %s"%d['B'])
        print()
