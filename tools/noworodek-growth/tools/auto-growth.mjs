import fs from "node:fs";
import { spawnSync } from "node:child_process";

// AIONS Noworodek: automatic, bounded capacity growth at the END of a training cycle.
// OWNER_APPROVED=2026-10-19; AUTHOR=Nez GPT-6; never writes from a second GPU job.
// Only MLP intermediate expands, only after independent fixed-benchmark plateau + A/B.
const H=process.env.HOME, LAB=`${H}/noworodek/growth-lab`;
const STATE=`${LAB}/auto-growth-state.json`, EVENTS=`${LAB}/auto-growth-events.jsonl`;
const GROW=`${H}/noworodek/gpu-trainer/target/debug/noworodek-grow`;
const DS=process.env.AIONS_NOWORODEK_SSH_ALIAS || "darkstar";
const GPU=process.env.AIONS_NOWORODEK_GPU_UUID;
if (!GPU && !process.argv.includes("--self-test")) throw Error("set AIONS_NOWORODEK_GPU_UUID before enabling growth");
const REMOTE="/garage/noworodek/gpu", BENCH=`${REMOTE}/lekcje-20261001.u16`;
const HOUR=3600000;
export function median(a){let b=[...a].sort((x,y)=>x-y);return b.length%2?b[(b.length-1)/2]:(b[b.length/2-1]+b[b.length/2])/2}
export function plateau(a) {
  const v=a.slice(-6);
  if(v.length<6 || Date.parse(v.at(-1).at)-Date.parse(v[0].at)<18*HOUR || v.some(x=>!Number.isFinite(x.loss)||x.loss<=0))return false;
  return median(v.slice(3).map(x=>x.loss)) >= median(v.slice(0,3).map(x=>x.loss))*0.995;
}
function cmd(command,args,t=120000){
  const r=spawnSync(command,args,{encoding:"utf8",timeout:t,maxBuffer:4<<20});
  if(r.status!==0||r.error)throw Error(`${command} exit=${r.status} ${r.error||""} ${(r.stderr||r.stdout||"").slice(-600)}`);
  return (r.stdout||"")+(r.stderr||"");
}
function ssh(c,t=120000){return cmd("ssh",["-o","BatchMode=yes","-o","ConnectTimeout=5",DS,c],t)}
function event(type,details={}){fs.appendFileSync(EVENTS,JSON.stringify({at:new Date().toISOString(),type,...details})+"\n",{mode:0o600})}
function persist(s){const p=STATE+".tmp";fs.writeFileSync(p,JSON.stringify(s,null,2)+"\n",{mode:0o600});fs.renameSync(p,STATE)}
function load(){
  if(fs.existsSync(STATE))return JSON.parse(fs.readFileSync(STATE,"utf8"));
  return {version:1,enabled:true,growthAt:"2026-10-19T15:34:55.000Z",lastObservationAt:null,lastAttemptAt:null,observations:[],promotions:1};
}
function modelInfo(file){
  const fd=fs.openSync(file,"r");let off=0,hidden=0,gate=0,count=0;
  const read=n=>{const b=Buffer.alloc(n);if(fs.readSync(fd,b,0,n,off)!==n)throw Error("short NWRD");off+=n;return b};
  try{
    if(read(4).toString()!=="NWRD"||read(4).readUInt32LE()!==1)throw Error("NWRD version mismatch");
    const nt=read(4).readUInt32LE();if(nt<10||nt>50)throw Error("tensor count mismatch");
    for(let k=0;k<nt;k++){
      const n=read(4).readUInt32LE();if(!n||n>1024)throw Error("bad name");
      const name=read(n).toString(),length=Number(read(8).readBigUInt64LE());
      if(!Number.isSafeInteger(length)||length>90_000_000)throw Error("bad tensor");
      if(name.endsWith("final_norm.weight"))hidden=length;
      if(name.endsWith("layers.00.mlp.gate_proj.weight"))gate=length;
      off+=length*4;count+=length;
    }
    if(off!==fs.fstatSync(fd).size||!hidden||gate%hidden!==0)throw Error("bad layout");
    return {width:gate/hidden,hidden,params:count,bytes:off};
  }finally{fs.closeSync(fd)}
}
function loss(out,kind){
  const match=(kind==="after"?/odlozone PO\s*:\s*([\d.]+)/:/odlozone po wczytaniu\s*:\s*([\d.]+)/).exec(out);
  const v=Number(match?.[1]);
  if(!match||!Number.isFinite(v)||v<=0)throw Error("missing loss");
  return v;
}
function gpuEval(folder,model,steps=0,seed=7){
  const command=`cd ${folder} && LD_LIBRARY_PATH=/opt/aions-cuda/lib CUDA_VISIBLE_DEVICES=${GPU} nice -n 10 timeout 200 ${REMOTE}/noworodek-gpu --ids ${BENCH} --load ${model} --steps ${steps} --batch ${steps?256:8} --lr 0.005 --seed ${seed} --eval-every 64 --eval-windows 256`;
  return loss(ssh(command,220000),steps?"after":"before");
}
function vramMiB(){
  const rows=ssh("nvidia-smi --query-gpu=uuid,memory.total --format=csv,noheader,nounits",30000).split("\n");
  const row=rows.find(r=>r.includes(GPU));
  const n=Number(row?.split(",").at(-1).trim());
  if(!Number.isFinite(n)||n<8000)throw Error("M60A not found by UUID");
  return n;
}
export function maybeGrow({dir,record,log=console.log}){
  // Explicitly dry-run without GPU, STOP, snapshots, or state mutations.
  if(process.env.AUTO_GROW_DRY_RUN==="1")return {status:"dry-run",model:modelInfo(`${dir}/best.bin`),state:load()};
  fs.mkdirSync(LAB,{recursive:true,mode:0o700});
  const s=load();const now=Date.now();
  if(!s.enabled||!record.accepted||fs.existsSync(`${dir}/STOP`))return {status:"not-eligible"};
  if((now-Date.parse(s.growthAt))<24*HOUR)return {status:"growth-cooldown"};
  if(s.lastObservationAt&&(now-Date.parse(s.lastObservationAt))<4*HOUR)return {status:"observation-not-due"};
  if(s.lastAttemptAt&&(now-Date.parse(s.lastAttemptAt))<24*HOUR)return {status:"attempt-cooldown"};
  const disk=fs.statfsSync(LAB),free=disk.bavail*disk.bsize;
  if(free<20*1024**3){event("blocked-disk",{free});return {status:"blocked-disk"}}
  const best=`${dir}/best.bin`,old=modelInfo(best),width=old.width*2;
  if(old.width<128||old.width%128||!Number.isSafeInteger(width))throw Error("invalid model width");
  const projected=old.bytes+(width-old.width)*old.hidden*3*4;
  const gpuCapacity=vramMiB()*1024**2;
  // Estimate weights, gradients, intermediate activations, logits and CUDA overhead.
  const estimatedPeak=projected*4 + 256*8*width*4*10 + 256*58296*4*2 + 512*1024**2;
  if(projected>gpuCapacity*0.10 || estimatedPeak>gpuCapacity*0.65){event("blocked-vram",{projected,estimatedPeak,gpuCapacity});return {status:"blocked-vram"}}
  // Sole writer at a cycle boundary: create immutable isolated copies first.
  const stamp=new Date().toISOString().replace(/[-:TZ.]/g,"").slice(0,14);
  const local=`${LAB}/auto-${stamp}`,remote=`/garage/noworodek/growth-lab/auto-growth-${stamp}`;
  fs.mkdirSync(local,{mode:0o700});
  const original=`${local}/before.bin.bak`,grown=`${local}/grown.bin`;
  fs.copyFileSync(best,original,fs.constants.COPYFILE_EXCL);
  let promoted=false;
  try{
    ssh(`mkdir -p ${remote}`,30000);
    cmd("scp",["-q",original,`${DS}:${remote}/before.bin.bak`]);
    const baseline=gpuEval(remote,"before.bin.bak");
    const obs={at:new Date().toISOString(),loss:baseline,width:old.width,cycle:record.cycle};
    s.observations.push(obs);s.observations=s.observations.slice(-16);s.lastObservationAt=obs.at;
    persist(s);event("observation",obs);
    log(`GROWTH fixed benchmark: ${baseline}, width ${old.width}; ${s.observations.length} observations`);
    if(!plateau(s.observations.filter(x=>x.width===old.width)))return {status:"watching",loss:baseline};
    if(fs.existsSync(`${dir}/STOP`))return {status:"stopped-by-operator"};
    s.lastAttemptAt=new Date().toISOString();persist(s);event("plateau",{from:old.width,to:width});
    cmd(GROW,["--source",original,"--output",grown,"--target-inter",String(width)],120000);
    const next=modelInfo(grown);
    if(next.width!==width||next.params<=old.params)throw Error("incorrect new parameter count");
    cmd("scp",["-q",grown,`${DS}:${remote}/grown.bin`]);
    const parity=gpuEval(remote,"grown.bin");
    if(parity.toFixed(4)!==baseline.toFixed(4)){
      event("rejected-parity",{baseline,parity});return {status:"rejected-parity"};
    }
    // Independent repeated-seed capacity gate; same immutable dataset and hyperparams.
    for(const seed of [7,19]){
      const before=gpuEval(remote,"before.bin.bak",64,seed);
      const after=gpuEval(remote,"grown.bin",64,seed);
      event("shadow-ab",{seed,before,after,width});
      if(!(after<before*0.995)){
        event("rejected-no-gain",{seed,before,after});return {status:"rejected-no-gain",seed};
      }
    }
    if(fs.existsSync(`${dir}/STOP`))return {status:"stopped-by-operator"};
    const stage=`${dir}/best.bin.growth-stage-${stamp}`;
    fs.copyFileSync(grown,stage,fs.constants.COPYFILE_EXCL);fs.renameSync(stage,best);promoted=true;
    s.promotions++;s.growthAt=new Date().toISOString();s.lastObservationAt=null;s.observations=[];
    persist(s);
    event("promoted",{oldParams:old.params,newParams:next.params,oldWidth:old.width,newWidth:width,original,grown});
    log(`GROWTH PROMOTED: ${old.params} -> ${next.params}, width ${old.width} -> ${width}`);
    return {status:"promoted",old:old.params,next:next.params};
  }catch(err){
    if(promoted){
      const revert=`${dir}/best.bin.rollback-stage-${stamp}`;
      fs.copyFileSync(original,revert);fs.renameSync(revert,best);event("rolled-back",{why:String(err)});
    }
    event("error",{cycle:record.cycle,why:String(err).slice(0,600)});throw err;
  }
}
if(process.argv.includes("--self-test")){
 const now=Date.now();
 const samples=[1.5,1.49,1.505,1.49,1.50,1.495].map((loss,i)=>({at:new Date(now-(5-i)*4*HOUR).toISOString(),loss}));
 if(!plateau(samples)||plateau(samples.slice(1))||plateau(samples.map((r,i)=>({...r,loss:2-i*.2}))))throw Error("plateau self-test");
 console.log("AUTO-GROW SELFTEST PASS");
}
