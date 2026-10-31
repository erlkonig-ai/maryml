"""Named GPU component oracle, NOT a whole official-model implementation.

Actual pinned layer2/layer7 weights; Torch static FP8 + scaled_mm, shipped
Triton Mamba core, and pinned HF eager-attention/RMSNorm cast boundaries.
No CPU recurrence, weight dequantization, downloads, or fallback backend.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct
import traceback

REV = 'da5b2650a835292c678994a1fa3fb653250fb508'
IMAGE = 'vllm/vllm-openai@sha256:3af90144a0926e5c5fe46ee16e5201e763dd854538b9d7ce433755f11dadaf78'
SHARDS = {2: (109705480, '18829df5916e1e388046848b4af922e7a15990957b93269a58924df4806069eb'),
          7: (35661352, '5ae1f86003d237659f6fb990ce55e2f4adeb9c0e81ac484b63de066b60746418')}
PLANS = {'T1': [('prefill',1)], 'T3': [('prefill',3)],
         'split': [('prefill',2),('step',1)], 'steps': [('step',1)]*3,
         'reset': [('prefill',3)]}
CAP = 32 << 20  # This half + native half <=64MiB raw tensor artifacts.
ARTIFACT_BOUND = 5*(4194304+61440)+13*(18560+10240+2*8192+2*4096)*2+4*3*4096*2+(2<<20)
BANDS = {'default': [0.02,0.02], 'norm': [0.002,0.02],
         'state': [0.002,0.02], 'conv_tail': [0.0,0.0]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Export:
    def __init__(self, root):
        self.root, self.used, self.records = root,0,[]

    def tensor(self, name, value):
        value = value.detach().contiguous().cpu()
        suffix = {torch.bfloat16:'bf16',torch.float32:'f32',torch.float8_e4m3fn:'e4m3fn'}[value.dtype]
        raw = value.view(torch.uint8).numpy().tobytes()
        if self.used+len(raw)>CAP or len(self.records)>=512:
            raise RuntimeError('32MiB/512 raw-artifact bound')
        path = self.root/(name+'.'+suffix)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise RuntimeError('duplicate artifact '+str(path))
        path.write_bytes(raw)
        self.used += len(raw)
        self.records.append(dict(path=str(path.relative_to(self.root)),shape=list(value.shape),
                                 dtype=suffix,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        write_json(self.root/'tensors.json',dict(byte_order='little',payload_bytes=self.used,tensors=self.records))
        if suffix != 'e4m3fn' and not torch.isfinite(value).all():
            raise RuntimeError('nonfinite output retained: '+name)


class Shard:
    def __init__(self, path, layer):
        self.file = path.open('rb')
        self.before = self.file_stat()
        size,digest = SHARDS[layer]
        assert self.before.st_size == size
        sha = hashlib.sha256()
        while data := self.file.read(1<<20):
            sha.update(data)
        assert sha.hexdigest() == digest, 'exact shard SHA256'
        self.file.seek(0)
        length = struct.unpack('<Q',self.file.read(8))[0]
        assert 0 < length < (1<<20)
        self.header = json.loads(self.file.read(length))
        self.base,self.layer = 8+length,layer
        self.record = dict(layer=layer,path=str(path),bytes=size,sha256=digest,tensors=[])

    def file_stat(self):
        import os
        return os.fstat(self.file.fileno())

    def tensor(self, suffix, dtype, shape):
        name = f'language_model.backbone.layers.{self.layer}.'+suffix
        info = self.header[name]
        assert info['dtype'] == dtype and info['shape'] == shape, (name,info,dtype,shape)
        a,b = info['data_offsets']
        width = {'BF16':2,'F8_E4M3':1,'F32':4}[dtype]
        count = 1
        for dim in shape:
            count *= dim
        assert b-a == count*width and 0 <= a <= b <= self.before.st_size-self.base
        self.file.seek(self.base+a)
        raw = self.file.read(b-a)
        assert len(raw) == b-a
        current = self.file_stat()
        assert (current.st_size,current.st_mtime_ns,current.st_ctime_ns) == (
            self.before.st_size,self.before.st_mtime_ns,self.before.st_ctime_ns)
        self.record['tensors'].append(dict(name=name,dtype=dtype,shape=shape,offset=self.base+a,bytes=len(raw)))
        tdtype = {'BF16':torch.bfloat16,'F8_E4M3':torch.float8_e4m3fn,'F32':torch.float32}[dtype]
        return torch.frombuffer(bytearray(raw),dtype=torch.uint8).view(tdtype).reshape(shape).clone().cuda()

    def operation(self, role, shape):
        base='mixer.'+role
        return (self.tensor(base+'.weight','F8_E4M3',shape),
                self.tensor(base+'.input_scale','F32',[1]),
                self.tensor(base+'.weight_scale','F32',[]).reshape(1))


def projection(x, operation, export=None, name=None):
    w,sa,sw = operation
    assert torch.isfinite(sa).all() and torch.isfinite(sw).all() and (sa>0).all() and (sw>0).all()
    flat=x.reshape(-1,x.shape[-1]).contiguous()
    q,returned = ops.scaled_fp8_quant(flat,scale=sa,num_token_padding=None,use_per_token_if_dynamic=False)
    assert q.dtype==torch.float8_e4m3fn and q.shape==flat.shape and returned.data_ptr()==sa.data_ptr()
    out=torch._scaled_mm(q,w.T,scale_a=sa,scale_b=sw,out_dtype=torch.bfloat16,
                         bias=None,scale_result=None,use_fast_accum=False).reshape(*x.shape[:-1],w.shape[0])
    if export is not None:
        export.tensor(name+'/input',x)
        export.tensor(name+'/qbytes',q)
        export.tensor(name+'/output',out)
    return out


def norm(x, gain):
    # Exact pinned HF eager cast order; reduction kernel is Torch's GPU kernel.
    wide=x.float()
    return gain*(wide*torch.rsqrt(wide.square().mean(-1,keepdim=True)+1e-5)).to(torch.bfloat16)


def run(args):
    global torch,ops
    assert ARTIFACT_BOUND < CAP
    versions={p:importlib.metadata.version(p) for p in ('torch','vllm','triton','flashinfer-python')}
    assert versions == {'torch':'2.11.0+cu129','vllm':'0.23.1rc1.dev1327+gf25953cc5.cu129',
                        'triton':'3.6.0','flashinfer-python':'0.6.14'}
    import torch
    from vllm import _custom_ops as ops
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,1)
    helper_raw=args.mamba_helper.read_bytes()
    assert hashlib.sha256(helper_raw).hexdigest()==args.helper_sha256
    spec=importlib.util.spec_from_file_location('retained_stock_mamba',args.mamba_helper)
    core=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)  # definitions only, no synthetic main execution
    core.torch=torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
    from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
    for fn in [causal_conv1d_fn,causal_conv1d_update,mamba_chunk_scan_combined_varlen,selective_state_update,rms_norm_gated]:
        setattr(core,fn.__name__,fn)
    for inventory, key in [(args.mamba_inventory,'sources'),(args.fp8_inventory,'files')]:
        for item in json.loads(inventory.read_text())[key]:
            path=Path('/usr/local/lib/python3.12/dist-packages')/item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], str(path)
    export=Export(args.out)
    t=torch.arange(3,dtype=torch.int64)[:,None]
    k=torch.arange(4096,dtype=torch.int64)[None,:]
    residual=(((17*t+13*k)%67-33).float()/16).to(torch.bfloat16).reshape(1,3,4096).cuda()
    export.tensor('residual',residual)
    report=dict(status='running',revision=REV,image=IMAGE,versions=versions,plans=PLANS,bands=BANDS,
                boundaries='BF16 norm before gain; static FP8; F32 recurrent state; no RoPE; BF16 outer sum',
                limits='Component GPU oracle, not official full-model execution. Diagnostic bands flag only.',sources=[])
    write_json(args.out/'report.json',report)
    source=Shard(args.mamba,2)
    gain=source.tensor('norm.weight','BF16',[4096])
    ip,op=source.operation('in_proj',[18560,4096]),source.operation('out_proj',[4096,8192])
    p={
        'conv_weight':source.tensor('mixer.conv1d.weight','BF16',[10240,1,4]).reshape(10240,4),
        'conv_bias':source.tensor('mixer.conv1d.bias','BF16',[10240]),
        'norm_gain':source.tensor('mixer.norm.weight','BF16',[8192]),
        'a_log':source.tensor('mixer.A_log','BF16',[128]).float(),
        'D':source.tensor('mixer.D','BF16',[128]).float(),
        'dt_bias':source.tensor('mixer.dt_bias','BF16',[128]).float()}
    p['A']=-torch.exp(p['a_log'])
    normalized=norm(residual,gain)
    export.tensor('mamba/norm',normalized)
    p['projected']=projection(normalized,ip,export,'diagnostic/mamba.in_proj')
    config=dict(name='mamba',batch=1,tokens=3,heads=128,head_dim=64,groups=8,state_dim=128,kernel=4,eps=1e-5)
    for arm,plan in PLANS.items():
        tokens=sum(count for _,count in plan)
        offset=0
        for part,(kind,count) in enumerate(plan):
            export.tensor(f'mamba/{arm}/part{part}-{kind}/projected',p['projected'][:,offset:offset+count])
            offset+=count
        # Bound state artifacts: only the final state/tail of each arm is retained.
        class Selected:
            def tensor(self,name,value):
                if name.endswith(('/state','/conv_tail')) and f'/part{len(plan)-1}-' not in name:
                    return
                export.tensor(name,value)
        result,_=core.run_arm(dict(config,tokens=tokens),p,plan,Selected(),arm)
        mixer=projection(result['output'],op,export if arm=='T3' else None,
                         'diagnostic/mamba.out_proj')
        export.tensor('mamba/'+arm+'/mixer',mixer)
        export.tensor('mamba/'+arm+'/residual',residual[:,:tokens]+mixer)
    report['sources'].append(source.record)
    source.file.close()
    del p,ip,op,gain,source
    source=Shard(args.attention,7)
    gain=source.tensor('norm.weight','BF16',[4096])
    projections=[source.operation(role,[n,4096]) for role,n in [('q_proj',4096),('k_proj',256),('v_proj',256),('o_proj',4096)]]
    normalized=norm(residual,gain)
    export.tensor('attention/norm',normalized)
    for arm,plan in PLANS.items():
        keys,values,start=[],[],0
        outputs=[]
        for part,(_,count) in enumerate(plan):
            prefix=f'attention/{arm}/part{part}'
            x=normalized[:,start:start+count].contiguous()
            q,k,v=[projection(x,p,export if arm=='T3' else None,'diagnostic/attention.'+role)
                   for p,role in zip(projections[:3],['q_proj','k_proj','v_proj'])]
            for label,value in [('q',q),('k',k),('v',v)]: export.tensor(prefix+'/'+label,value)
            keys.append(k); values.append(v)
            kh=torch.cat(keys,1).view(1,-1,2,128).transpose(1,2).repeat_interleave(16,dim=1)
            vh=torch.cat(values,1).view(1,-1,2,128).transpose(1,2).repeat_interleave(16,dim=1)
            qh=q.view(1,count,32,128).transpose(1,2)
            scores=(qh@kh.transpose(2,3))*(128**-0.5)
            mask=torch.arange(start+count,device='cuda')[None,:]>torch.arange(start,start+count,device='cuda')[:,None]
            scores=scores.masked_fill(mask,float('-inf'))
            probs=torch.softmax(scores,dim=-1,dtype=torch.float32).to(torch.bfloat16)
            context=(probs@vh).transpose(1,2).contiguous().reshape(1,count,4096)
            export.tensor(prefix+'/attention',context)
            out=projection(context,projections[3],export if arm=='T3' else None,'diagnostic/attention.o_proj')
            export.tensor(prefix+'/mixer',out)
            export.tensor(prefix+'/residual',residual[:,start:start+count]+out)
            outputs.append(out); start+=count
        export.tensor('attention/'+arm+'/keys',torch.cat(keys,1))
        export.tensor('attention/'+arm+'/values',torch.cat(values,1))
    report['sources'].append(source.record)
    source.file.close()
    torch.cuda.synchronize()
    report.update(status='complete',payload_bytes=export.used,
                  torch_peak_allocated=torch.cuda.max_memory_allocated(),torch_peak_reserved=torch.cuda.max_memory_reserved())
    write_json(args.out/'report.json',report)
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['mamba','attention','mamba-helper','mamba-inventory','fp8-inventory','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--helper-sha256',required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args)
    except BaseException:
        (args.out/'failure.txt').write_text(traceback.format_exc())
        raise
