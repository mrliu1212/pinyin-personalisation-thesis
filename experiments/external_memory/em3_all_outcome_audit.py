from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean, median

AUTHORS = ["Etinjat", "Re_spectators", "breaddddd"]
GROUPS = [
    "G_OK_F_OK_H_OK","G_OK_F_OK_H_BAD","G_OK_F_BAD_H_OK","G_OK_F_BAD_H_BAD",
    "G_BAD_F_OK_H_OK","G_BAD_F_OK_H_BAD","G_BAD_F_BAD_H_OK","G_BAD_F_BAD_H_BAD",
]
FOCUS = [
    "G_OK_H_BAD","F_OK_H_BAD","G_OK_F_OK_H_BAD","G_OK_F_BAD_H_BAD",
    "G_BAD_F_OK_H_BAD","G_BAD_F_OK_H_OK",
    "G_BAD_F_BAD_H_BAD_GOLD_IN_HISTORY",
    "G_BAD_F_BAD_H_BAD_GOLD_NOT_IN_HISTORY",
    "G_BAD_F_BAD_H_OK_GOLD_IN_HISTORY",
    "G_BAD_F_BAD_H_OK_GOLD_NOT_IN_HISTORY",
]

def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def pick(r,*keys,default=None):
    for k in keys:
        if k in r: return r[k]
    return default

def rid(r): return str(pick(r,"row_id","condition_id","id"))
def auth(r): return str(pick(r,"author","user_id"))
def gold(r): return str(pick(r,"gold","target","target_candidate","current_gold"))
def ctx(r): return str(pick(r,"context","preceding_context","current_context",default="") or "")
def doc(r):
    x=pick(r,"work_id","document_id","source_work_id","page_id","source_id","work_key","article_id")
    return None if x is None else str(x)
def pos(r,fallback):
    x=pick(r,"chronological_position","position","interaction_position","query_position")
    return fallback if x is None else int(x)
def py(r):
    x=pick(r,"pinyin_segments","segmented_pinyin","pinyin")
    return tuple(map(str,x)) if isinstance(x,(list,tuple)) else (str(x),)
def top1(x):
    try: return int(x)==1
    except (TypeError,ValueError): return False
def group(g,f,h):
    return f"G_{'OK' if g else 'BAD'}_F_{'OK' if f else 'BAD'}_H_{'OK' if h else 'BAD'}"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def entropy(c):
    n=sum(c.values())
    return 0.0 if not n else -sum((v/n)*math.log2(v/n) for v in c.values() if v)
def rank(cands,lf,lc):
    return sorted(cands,key=lambda c:(-(float(c["normalized_generic_score"])+lf*float(c["frequency_support"])+lc*float(c["context_support"])),int(c["generic_rank"])))
def brief(s,n=220):
    s=" ".join(str(s).replace("\n"," ").split())
    return s[-n:]
def stat(vals):
    vals=sorted(float(x) for x in vals)
    if not vals:return None
    def q(p):
        if len(vals)==1:return vals[0]
        x=(len(vals)-1)*p;a=int(x);b=math.ceil(x)
        return vals[a] if a==b else vals[a]*(b-x)+vals[b]*(x-a)
    return {"n":len(vals),"mean":mean(vals),"median":median(vals),"min":vals[0],
            "p25":q(.25),"p75":q(.75),"p90":q(.9),"max":vals[-1]}
def pct(a,b): return 0.0 if not b else a/b

def summarize(rs):
    n=len(rs)
    pats=Counter((r["author"],r["pinyin"],r["gold"]) for r in rs)
    docs=Counter((r["author"],r["document_id"]) for r in rs if r["document_id"])
    t3=sum(v for _,v in docs.most_common(3))
    return {
        "rows":n,
        "authors":dict(Counter(r["author"] for r in rs)),
        "gold_in_history":{"count":sum(r["gold_in_history"] for r in rs),"share":pct(sum(r["gold_in_history"] for r in rs),n)},
        "gold_in_candidates":{"count":sum(r["gold_in_candidates"] for r in rs),"share":pct(sum(r["gold_in_candidates"] for r in rs),n)},
        "raw_winner_is_gold":{"count":sum(r["raw_winner_is_gold"] for r in rs),"share":pct(sum(r["raw_winner_is_gold"] for r in rs),n)},
        "same_pinyin_history":stat(r["same_pinyin_history"] for r in rs),
        "raw_winner_share":stat(r["raw_winner_share"] for r in rs),
        "raw_margin":stat(r["raw_margin"] for r in rs),
        "distinct_targets":stat(r["distinct_targets"] for r in rs),
        "entropy_bits":stat(r["entropy_bits"] for r in rs),
        "gold_count_oracle":stat(r["gold_count"] for r in rs),
        "gold_share_oracle":stat(r["gold_share"] for r in rs),
        "unique_author_pinyin_gold":len(pats),
        "largest_pattern":max(pats.values(),default=0),
        "unique_author_documents":len(docs),
        "top3_document_rows":t3,
        "top3_document_share":pct(t3,n),
        "top_patterns":[{"author":k[0],"pinyin":k[1],"gold":k[2],"count":v} for k,v in pats.most_common(8)],
        "top_documents":[{"author":k[0],"document_id":k[1],"count":v} for k,v in docs.most_common(8)],
    }

def choose(rs,n):
    out=[];seen=set()
    for r in sorted(rs,key=lambda x:(-x["same_pinyin_history"],-x["raw_winner_share"],x["row_id"])):
        k=(r["author"],r["pinyin"],r["gold"])
        if k not in seen:
            out.append(r);seen.add(k)
            if len(out)>=n:return out
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot-root",type=Path,default=Path(r"C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory"))
    ap.add_argument("--four-way",type=Path,default=Path(r"C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_four_way_dev_compare\rows.jsonl"))
    ap.add_argument("--surface",type=Path,default=Path(r"C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_fixed_gfc_dev\selected_rows.jsonl"))
    ap.add_argument("--output",type=Path,default=Path(r"results\personalisation\external_memory\em3_all_outcome_audit"))
    ap.add_argument("--authors",nargs="+",default=AUTHORS)
    ap.add_argument("--history-budget",type=int,default=5000)
    ap.add_argument("--examples",type=int,default=4)
    a=ap.parse_args()

    hp=a.pilot_root/"history_manifest.jsonl"; dp=a.pilot_root/"dev_manifest.jsonl"
    src={"history_manifest":hp,"dev_manifest":dp,"four_way_rows":a.four_way,"surface_rows":a.surface}
    for k,p in src.items():
        if not p.exists(): raise FileNotFoundError(f"{k}: {p}")
        if "test" in p.name.lower(): raise RuntimeError(f"possible Test input refused: {p}")

    four={rid(r):r for r in rows(a.four_way)}
    surf={rid(r):r for r in rows(a.surface)}
    aset=set(a.authors)
    dev=[r for r in rows(dp) if auth(r) in aset and rid(r) in four]

    by=defaultdict(list);i=0
    for r in rows(hp):
        if auth(r) in aset: by[auth(r)].append((pos(r,i),i,False,r))
        i+=1
    for r in dev:
        by[auth(r)].append((pos(r,i),i,True,r));i+=1

    allr=[];gr={g:[] for g in GROUPS};validation=Counter()
    for au in a.authors:
        data=sorted(by[au],key=lambda x:(x[0],x[1]));vis=deque(maxlen=a.history_budget);i=0
        while i<len(data):
            p0=data[i][0];j=i;block=[]
            while j<len(data) and data[j][0]==p0:block.append(data[j]);j+=1
            for qp,_,isdev,q in block:
                qid=rid(q)
                if not isdev or qid not in four:continue
                z=four[qid];go=top1(z.get("G_rank"));fo=top1(z.get("F_rank"));ho=top1(z.get("Hidden_M1_rank"))
                qpy=py(q);qg=gold(q);visible=[x for _,x in vis];same=[x for x in visible if py(x)==qpy]
                cnt=Counter(gold(x) for x in same);dist=cnt.most_common()
                rw=dist[0][0] if dist else None; rwc=dist[0][1] if dist else 0; r2=dist[1][1] if len(dist)>1 else 0
                gc=cnt.get(qg,0); gh_rank=next((k for k,(t,_) in enumerate(dist,1) if t==qg),None)
                gp=fp=hp_=None;cset=[];evidence=[]
                sr=surf.get(qid)
                if sr and isinstance(sr.get("ranking"),list) and sr["ranking"]:
                    cs=sr["ranking"];cset=[str(c["candidate"]) for c in cs]
                    gp=str(rank(cs,0,0)[0]["candidate"]);fp=str(rank(cs,4,0)[0]["candidate"]);hp_=str(rank(cs,0,4)[0]["candidate"])
                    validation["n"]+=1;validation["G"]+=((gp==qg)==go);validation["F"]+=((fp==qg)==fo);validation["H"]+=((hp_==qg)==ho)
                    for c in rank(cs,0,4)[:6]:
                        evidence.append({"candidate":str(c["candidate"]),"generic_rank":int(c["generic_rank"]),
                          "Gscore":float(c["normalized_generic_score"]),"freq":float(c["frequency_support"]),
                          "ctx":float(c["context_support"]),"Hscore":float(c["normalized_generic_score"])+4*float(c["context_support"])})
                rec={
                    "row_id":qid,"group":group(go,fo,ho),"author":au,"document_id":doc(q),"query_position":qp,
                    "pinyin":" ".join(qpy),"gold":qg,"G":gp,"F":fp,"Hidden_M1":hp_,
                    "G_correct":go,"F_correct":fo,"H_correct":ho,
                    "visible_history":len(visible),"same_pinyin_history":len(same),"distinct_targets":len(cnt),"entropy_bits":entropy(cnt),
                    "raw_winner":rw,"raw_winner_count":rwc,"raw_winner_share":rwc/len(same) if same else 0.0,
                    "raw_second_count":r2,"raw_margin":rwc-r2,"target_distribution":dist[:20],
                    "gold_in_history":gc>0,"gold_count":gc,"gold_share":gc/len(same) if same else 0.0,"gold_history_rank":gh_rank,
                    "gold_in_candidates":qg in cset,"raw_winner_is_gold":rw==qg if rw else False,
                    "G_equals_raw_winner":gp==rw if gp else False,"F_equals_raw_winner":fp==rw if fp else False,"H_equals_raw_winner":hp_==rw if hp_ else False,
                    "current_context":brief(ctx(q),260),"candidate_evidence":evidence,
                }
                allr.append(rec);gr[rec["group"]].append(rec)
            for pp,_,_,r in block:vis.append((pp,r))
            i=j

    focus={
      "G_OK_H_BAD":[r for r in allr if r["G_correct"] and not r["H_correct"]],
      "F_OK_H_BAD":[r for r in allr if r["F_correct"] and not r["H_correct"]],
      "G_OK_F_OK_H_BAD":gr["G_OK_F_OK_H_BAD"],
      "G_OK_F_BAD_H_BAD":gr["G_OK_F_BAD_H_BAD"],
      "G_BAD_F_OK_H_BAD":gr["G_BAD_F_OK_H_BAD"],
      "G_BAD_F_OK_H_OK":gr["G_BAD_F_OK_H_OK"],
      "G_BAD_F_BAD_H_BAD_GOLD_IN_HISTORY":[r for r in gr["G_BAD_F_BAD_H_BAD"] if r["gold_in_history"]],
      "G_BAD_F_BAD_H_BAD_GOLD_NOT_IN_HISTORY":[r for r in gr["G_BAD_F_BAD_H_BAD"] if not r["gold_in_history"]],
      "G_BAD_F_BAD_H_OK_GOLD_IN_HISTORY":[r for r in gr["G_BAD_F_BAD_H_OK"] if r["gold_in_history"]],
      "G_BAD_F_BAD_H_OK_GOLD_NOT_IN_HISTORY":[r for r in gr["G_BAD_F_BAD_H_OK"] if not r["gold_in_history"]],
    }

    a.output.mkdir(parents=True,exist_ok=True);(a.output/"groups").mkdir(exist_ok=True);(a.output/"focused_subsets").mkdir(exist_ok=True)
    def dump(path,rs):
        with path.open("w",encoding="utf-8",newline="\n") as f:
            for r in rs:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    dump(a.output/"all_rows.jsonl",allr)
    for k,v in gr.items():dump(a.output/"groups"/f"{k.lower()}.jsonl",v)
    for k,v in focus.items():dump(a.output/"focused_subsets"/f"{k.lower()}.jsonl",v)

    summary={
      "schema_version":1,"experiment":"em3_all_outcome_audit",
      "condition":"Full+Short / H5000 / old exploratory 3-author Dev diagnostic surface",
      "authors":a.authors,"test_used":False,"rows":len(allr),
      "prediction_validation":{"n":validation["n"],"G_match":pct(validation["G"],validation["n"]),"F_match":pct(validation["F"],validation["n"]),"H_match":pct(validation["H"],validation["n"])},
      "all_groups":{k:summarize(gr[k]) for k in GROUPS},
      "focused_subsets":{k:summarize(focus[k]) for k in FOCUS},
    }
    (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    prov={"test_used":False,"authors":a.authors,"history_budget":a.history_budget,
          "inputs":{k:{"path":str(p.resolve()),"sha256":sha(p)} for k,p in src.items()},
          "reproduce":r"$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'"+ "\n"+r"& $python -m experiments.external_memory.em3_all_outcome_audit"}
    (a.output/"provenance.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    L=[];w=L.append
    w("#"*110);w("EM3 ALL OUTCOME AUDIT");w("Test used: NO");w(f"Rows: {len(allr)}");w("")
    w("ALL 8 G/F/H GROUPS")
    for k in GROUPS:
        s=summary["all_groups"][k]
        w(f"{k:28s} N={s['rows']:4d} | Gold-in-history={s['gold_in_history']['share']:.1%} | raw-winner=Gold={s['raw_winner_is_gold']['share']:.1%} | Gold-in-candidates={s['gold_in_candidates']['share']:.1%}")
    w("");w("FOCUSED SUBSETS")
    for k in FOCUS:
        s=summary["focused_subsets"][k]
        w(f"\n{k}  N={s['rows']}")
        w(f" authors={s['authors']}")
        w(f" Gold-in-history={s['gold_in_history']['share']:.1%}; Gold-in-candidates={s['gold_in_candidates']['share']:.1%}; raw-winner=Gold={s['raw_winner_is_gold']['share']:.1%}")
        w(f" same-Pinyin={s['same_pinyin_history']}")
        w(f" raw-winner-share={s['raw_winner_share']}")
        w(f" raw-margin={s['raw_margin']}")
        w(f" distinct-targets={s['distinct_targets']}")
        w(f" entropy={s['entropy_bits']}")
        w(f" top3-doc-share={s['top3_document_share']:.1%}")
        w(f" top-patterns={s['top_patterns'][:5]}")
        w(f" top-docs={s['top_documents'][:5]}")

    exgroups=["G_OK_F_OK_H_BAD","G_OK_F_BAD_H_BAD","G_BAD_F_OK_H_BAD","G_BAD_F_OK_H_OK",
              "G_BAD_F_BAD_H_BAD_GOLD_IN_HISTORY","G_BAD_F_BAD_H_OK_GOLD_IN_HISTORY"]
    w("");w("="*110);w("EXAMPLES")
    for k in exgroups:
        w("\n"+"#"*110);w(f"{k} | TOTAL={len(focus[k])}")
        for r in choose(focus[k],a.examples):
            w("\n"+"-"*110)
            w(f"ROW={r['row_id']} | AUTHOR={r['author']} | DOC={r['document_id']}")
            w(f"PINYIN={r['pinyin']} | GOLD[analysis]={r['gold']} | G={r['G']} | F={r['F']} | H={r['Hidden_M1']}")
            w(f"VISIBLE={r['visible_history']} | SAME-PINYIN={r['same_pinyin_history']} | DISTINCT={r['distinct_targets']} | ENTROPY={r['entropy_bits']:.3f}")
            w(f"RAW WINNER={r['raw_winner']} {r['raw_winner_count']}/{r['same_pinyin_history']} share={r['raw_winner_share']:.3f} margin={r['raw_margin']}")
            w(f"DISTRIBUTION={r['target_distribution']}")
            w(f"GOLD-IN-HISTORY[analysis]={r['gold_in_history']} count={r['gold_count']} share={r['gold_share']:.3f} rank={r['gold_history_rank']}")
            w(f"GOLD-IN-CANDIDATES={r['gold_in_candidates']}")
            w("CONTEXT: "+r["current_context"])
            w("CANDIDATE EVIDENCE:")
            for c in r["candidate_evidence"]:
                w(f"  {c['candidate']} | G-rank={c['generic_rank']} | G={c['Gscore']:.3f} | freq={c['freq']:.3f} | ctx={c['ctx']:.3f} | H={c['Hscore']:.3f}")

    w("\n"+"="*110);w("REPRODUCIBILITY")
    for k,v in prov["inputs"].items():w(f"{k}: {v['path']}\n  sha256={v['sha256']}")
    w("\nReproduce:\n"+prov["reproduce"])
    report="\n".join(L)+"\n"
    (a.output/"report.txt").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()
