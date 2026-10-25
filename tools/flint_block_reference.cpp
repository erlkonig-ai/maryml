// Independent CPU F32/math reference for isolated Flint blocks 0/2/5; no quantized runtime parity.
// Successor of retained layer2-reference.cpp (a3b3d5ad...); sparse expert selection
// is computed by GGML, then mapped explicitly from global IDs to local slices.
// Graph follows llama.cpp 1ddffe9f70a63e2d260d4bfad9ad200009007f62:
// src/models/flint.cpp and src/llama-graph.cpp. Native GGML performs all operators
// and NVFP4 decoding. No tokenizer, model loader, CUDA registration or server.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <sys/stat.h>

extern "C" void dequantize_row_nvfp4(const void *, float *, int64_t);
namespace fs = std::filesystem;
using Tensor = ggml_tensor;
static void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
static std::string digest(const void *p, size_t n) {
    unsigned char h[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char *>(p), n, h);
    static const char hex[] = "0123456789abcdef";
    std::string s;
    for (unsigned char b : h) { s += hex[b >> 4]; s += hex[b & 15]; }
    return s;
}
static void write_bytes(const fs::path &p, const void *data, size_t size) {
    require(!fs::exists(p), "refusing overwrite: " + p.string());
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char *>(data), size);
    f.close();
    require(bool(f), "write failed: " + p.string());
}
struct Graph {
    ggml_context *ctx;
    ggml_cgraph *gf;
    std::vector<std::pair<std::string, Tensor *>> outputs;
    Graph() {
        ctx = ggml_init({256ull << 20, nullptr, false});
        require(ctx, "graph allocation failed");
        gf = ggml_new_graph_custom(ctx, 4096, false);
    }
    ~Graph() { ggml_free(ctx); }
    Tensor *mark(const std::string &name, Tensor *t) {
        ggml_set_name(t, name.c_str());
        outputs.emplace_back(name, t);
        return t;
    }
    Tensor *input(const std::string &name, int64_t n0, int64_t n1, const std::vector<float> &v) {
        require(v.size() == size_t(n0*n1), "input shape mismatch");
        auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        memcpy(t->data, v.data(), v.size()*4);
        return mark(name, t);
    }
    void run() {
        for (auto &[name, t] : outputs) ggml_build_forward_expand(gf, t);
        const auto rc = ggml_graph_compute_with_ctx(ctx, gf, 4);
        require(rc == GGML_STATUS_SUCCESS, "GGML compute failed: " + std::to_string(rc));
    }
    void dump(const fs::path &out) {
        require(fs::create_directory(out), "output directory already exists");
        std::ofstream index(out / "tensors.tsv");
        index << "name\ttype\tne0\tne1\tne2\tne3\tbytes\tsha256\tmin\tmax\trms\n";
        for (auto &[name, t] : outputs) {
            require(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32, "unsupported readback type");
            std::vector<unsigned char> bytes(ggml_nelements(t)*4);
            size_t dst = 0;
            double sumsq = 0;
            double lo = INFINITY, hi = -INFINITY;
            for (int64_t i3=0; i3<t->ne[3]; ++i3)
            for (int64_t i2=0; i2<t->ne[2]; ++i2)
            for (int64_t i1=0; i1<t->ne[1]; ++i1)
            for (int64_t i0=0; i0<t->ne[0]; ++i0) {
                const auto *p = static_cast<const char *>(t->data) + i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
                memcpy(bytes.data()+dst, p, 4); dst += 4;
                float f; int32_t i;
                memcpy(&f, p, 4); memcpy(&i, p, 4);
                double x = t->type == GGML_TYPE_F32 ? double(f) : double(i);
                require(std::isfinite(x) || name == "attention_mask", "nonfinite output: " + name);
                lo = std::min(lo, x); hi = std::max(hi, x); sumsq += x*x;
            }
            const std::string ext = t->type == GGML_TYPE_F32 ? ".f32" : ".i32";
            write_bytes(out/(name+ext), bytes.data(), bytes.size());
            index << name << '\t' << ggml_type_name(t->type);
            for (auto n : t->ne) index << '\t' << n;
            index << '\t' << bytes.size() << '\t' << digest(bytes.data(), bytes.size())
                  << '\t' << std::setprecision(17) << lo << '\t' << hi << '\t' << sqrt(sumsq/ggml_nelements(t)) << '\n';
        }
        index.close(); require(bool(index), "tensor index write failed");
    }
};

struct Weights {
    ggml_context *ctx;
    gguf_context *header;
    int block;
    bool rope;
    bool swa;
    std::vector<int32_t> local_selection;
    struct stat identity;
    std::ofstream selected_receipt;
    FILE *file;
    std::map<std::string, Tensor *> tensors; // ephemeral reference operation only
    std::ofstream receipt;
    Weights(const std::string &path, int block_id, const fs::path &out) : block(block_id) {
        struct stat st;
        require(stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode), "model is not regular");
        require(block == 0 || block == 2 || block == 5, "only block 0/2/5");
        require(st.st_size == 22863084160ll && st.st_dev == 66306 && st.st_ino == 32399645
            && st.st_mtim.tv_sec == 1792849073 && st.st_mtim.tv_nsec == 87120678, "model identity changed");
        identity = st;
        header = gguf_init_from_file(path.c_str(), {true, nullptr});
        require(header, "GGUF index failed");
        require(gguf_get_data_offset(header) == 5512960 && gguf_get_n_tensors(header) == 1264, "header identity changed");
        auto key = [&](const std::string &name, gguf_type type) {
            const auto id = gguf_find_key(header,name.c_str());
            require(id >= 0 && gguf_get_kv_type(header,id) == type,"metadata type/missing: "+name);
            return id;
        };
        require(std::string(gguf_get_val_str(header,key("general.architecture",GGUF_TYPE_STRING))) == "flint","architecture differs");
        for(const auto &[name,value] : std::vector<std::pair<std::string,uint32_t>>{
            {"block_count",49},{"context_length",262144},{"embedding_length",2048},{"feed_forward_length",6144},
            {"attention.head_count",32},{"attention.head_count_kv",4},
            {"attention.key_length",128},{"attention.value_length",128},
            {"leading_dense_block_count",2},{"expert_count",256},{"expert_used_count",6},
            {"expert_gating_func",2},{"expert_feed_forward_length",512},
            {"expert_shared_feed_forward_length",1024},{"expert_shared_count",2},{"attention.sliding_window",513}})
            require(gguf_get_val_u32(header,key("flint."+name,GGUF_TYPE_UINT32)) == value,"metadata value: "+name);
        for(const auto &[name,value] : std::vector<std::pair<std::string,float>>{
            {"rope.freq_base",10000.0f},{"attention.layer_norm_rms_epsilon",1e-6f},{"expert_weights_scale",1.0f}})
            require(gguf_get_val_f32(header,key("flint."+name,GGUF_TYPE_FLOAT32)) == value,"metadata value: "+name);
        require(gguf_get_val_bool(header,key("flint.expert_weights_norm",GGUF_TYPE_BOOL)),"expert norm differs");
        auto flag = [&](const std::string &name) {
            const auto id = key(name,GGUF_TYPE_ARRAY);
            require(gguf_get_arr_type(header,id) == GGUF_TYPE_BOOL && gguf_get_arr_n(header,id) == 49,"metadata pattern: "+name);
            return static_cast<const bool *>(gguf_get_arr_data(header,id))[block];
        };
        rope = flag("flint.attention.rope_pattern");
        swa = flag("flint.attention.sliding_window_pattern");
        require(rope == (block == 2) && swa == (block == 2),"fixture metadata semantics differ");
        std::cout << "metadata_bound=true block=" << block << " rope=" << rope << " swa=" << swa << std::endl;
        file = fopen(path.c_str(), "rb"); require(file, "model open failed");
        ctx = ggml_init({512ull << 20, nullptr, false}); require(ctx, "weight arena allocation failed");
        require(!fs::exists(out), "tensor receipt already exists");
        receipt.open(out);
        selected_receipt.open(out.parent_path()/"selected-experts.tsv");
        selected_receipt << "local_slice\ttoken\tslot\tglobal_id\n";
        receipt << "name\tstored_type\tne0\tne1\tne2\tne3\tabsolute_offset\tstored_bytes\traw_sha256\tdecoded_f32_sha256\n";
        std::cout << "model_path=" << path << " size=" << st.st_size << " device=" << st.st_dev
                  << " inode=" << st.st_ino << " mtime=" << st.st_mtim.tv_sec << '.' << st.st_mtim.tv_nsec
                  << " data_offset=" << gguf_get_data_offset(header) << '\n';
    }
    ~Weights() { fclose(file); gguf_free(header); ggml_free(ctx); }
    void check_identity() {
        struct stat st;
        require(fstat(fileno(file),&st) == 0 && st.st_size == identity.st_size
            && st.st_dev == identity.st_dev && st.st_ino == identity.st_ino
            && st.st_mtim.tv_sec == identity.st_mtim.tv_sec && st.st_mtim.tv_nsec == identity.st_mtim.tv_nsec,
            "model changed during reference");
    }
    size_t absolute_offset(int64_t tensor) {
        const size_t data=gguf_get_data_offset(header), relative=gguf_get_tensor_offset(header,tensor);
        require(data<=size_t(identity.st_size) && relative<=size_t(identity.st_size)-data,"tensor offset outside model");
        return data+relative;
    }
    Tensor *load(const std::string &stem, std::array<int64_t,4> expected, ggml_type type) {
        const std::string name = "blk." + std::to_string(block) + "." + stem;
        const int64_t id = gguf_find_tensor(header, name.c_str());
        require(id >= 0, "missing tensor: " + name);
        require(gguf_get_tensor_type(header,id) == type, "stored type mismatch: " + name);
        const auto *ne = gguf_get_tensor_ne(header,id);
        for (int j=0;j<4;++j) require(ne[j] == expected[j], "shape mismatch: " + name);
        const size_t size = gguf_get_tensor_size(header,id);
        const size_t off = absolute_offset(id);
        require(off <= 22863084160ull && size <= 22863084160ull-off, "tensor extent outside model");
        require(size <= (16ull << 20), "reference read exceeds bounded dense/slice limit");
        std::vector<unsigned char> raw(size);
        require(fseeko(file, off, SEEK_SET) == 0 && fread(raw.data(),1,size,file) == size, "tensor read failed: " + name);
        auto *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0],ne[1],ne[2],ne[3]);
        ggml_set_name(t,name.c_str());
        const auto n = ggml_nelements(t);
        auto *dst = static_cast<float *>(t->data);
        if (type == GGML_TYPE_NVFP4) dequantize_row_nvfp4(raw.data(),dst,n);
        else if (type == GGML_TYPE_BF16) ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(raw.data()),dst,n);
        else { require(size == size_t(n)*4, "F32 size mismatch"); memcpy(dst,raw.data(),size); }
        for (int64_t i=0;i<n;++i) require(std::isfinite(dst[i]), "nonfinite stored tensor: " + name);
        receipt << name << '\t' << ggml_type_name(type);
        for (auto d : expected) receipt << '\t' << d;
        receipt << '\t' << off << '\t' << size << '\t' << digest(raw.data(),raw.size())
                << '\t' << digest(dst,n*4) << '\n'; receipt.flush();
        require(bool(receipt), "tensor receipt write failed");
        tensors.emplace(stem,t);
        std::cout << "loaded " << name << " raw_bytes=" << size << std::endl;
        return t;
    }
    // Compact unique global experts in first-seen order, retaining all original
    // token/slot selections. Repeated global IDs share the same local ID, so
    // GGML's per-expert row grouping matches the original full-bank graph.
    // Original IDs remain the trace and gather the original 256-element scales.
    std::vector<int32_t> load_selected(Tensor *ids, int T) {
        std::vector<int32_t> global(6*T);
        std::vector<int32_t> unique;
        for(int t=0;t<T;++t) for(int slot=0;slot<6;++slot) {
            int32_t id;
            memcpy(&id,static_cast<char *>(ids->data)+t*ids->nb[1]+slot*ids->nb[0],4);
            require(id >= 0 && id < 256,"invalid GGML-selected global ID");
            global[t*6+slot]=id;
            auto found=std::find(unique.begin(),unique.end(),id);
            const int32_t local=found-unique.begin();
            if(found==unique.end()) unique.push_back(id);
            local_selection.push_back(local);
            selected_receipt << local << '\t' << t << '\t' << slot << '\t' << id << '\n';
        }
        selected_receipt.flush(); require(bool(selected_receipt),"selected ID receipt failed");
        for(const auto &stem : {"ffn_up_exps","ffn_gate_exps","ffn_down_exps"}) {
            const bool down=std::string(stem)=="ffn_down_exps";
            const int64_t ni=down?512:2048, no=down?2048:512;
            const std::string suffix=std::string(stem)+".weight";
            const std::string name="blk."+std::to_string(block)+"."+suffix;
            const int64_t tensor=gguf_find_tensor(header,name.c_str());
            require(tensor >= 0 && gguf_get_tensor_type(header,tensor)==GGML_TYPE_NVFP4,"expert type missing");
            const auto *ne=gguf_get_tensor_ne(header,tensor);
            require(ne[0]==ni && ne[1]==no && ne[2]==256 && ne[3]==1,"expert shape differs");
            const size_t row_bytes=size_t(ni/64)*36, size=row_bytes*size_t(no);
            require(size==589824 && gguf_get_tensor_size(header,tensor)==size*256,"expert extent differs");
            const size_t base=absolute_offset(tensor);
            require(size*256<=size_t(identity.st_size)-base,"expert bank extent outside model");
            auto *weight=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,ni,no,unique.size());
            for(size_t local=0;local<unique.size();++local) {
                const size_t off=base+size*size_t(unique[local]);
                require(off <= 22863084160ull && size <= 22863084160ull-off,"expert slice outside model");
                std::vector<unsigned char> raw(size);
                require(fseeko(file,off,SEEK_SET)==0 && fread(raw.data(),1,size,file)==size,"expert slice read failed");
                auto *dst=static_cast<float *>(weight->data)+local*ni*no;
                dequantize_row_nvfp4(raw.data(),dst,ni*no);
                for(int64_t i=0;i<ni*no;++i) require(std::isfinite(dst[i]),"nonfinite expert");
                receipt << name << "[global=" << unique[local] << ",local=" << local << "]\tnvfp4\t"
                    << ni << '\t' << no << "\t1\t1\t" << off << '\t' << size << '\t'
                    << digest(raw.data(),raw.size()) << '\t' << digest(dst,ni*no*4) << '\n';
            }
            tensors.emplace(suffix,weight);
        }
        receipt.flush(); require(bool(receipt),"expert receipt failed"); check_identity();
        return global;
    }
    Tensor *at(const std::string &name) { return tensors.at(name); }
    void all() {
        for (const auto &s : {"attn_norm.weight", "ffn_norm.weight"}) load(s,{2048,1,1,1},GGML_TYPE_F32);
        for (const auto &s : {"attn_q_norm.weight", "attn_k_norm.weight"}) load(s,{128,1,1,1},GGML_TYPE_F32);
        load("attn_sinks.weight",{32,1,1,1},GGML_TYPE_F32);
        std::vector<std::pair<std::string,std::array<int64_t,4>>> mats = {
            {"attn_q",{2048,4096,1,1}}, {"attn_k",{2048,512,1,1}},
            {"attn_v",{2048,512,1,1}}, {"attn_output",{4096,2048,1,1}}};
        if(block==0) {
            mats.insert(mats.end(), {{"ffn_gate",{2048,6144,1,1}},
                {"ffn_up",{2048,6144,1,1}}, {"ffn_down",{6144,2048,1,1}}});
        } else {
            load("ffn_gate_inp.weight",{2048,256,1,1},GGML_TYPE_BF16);
            mats.insert(mats.end(), {
            {"ffn_gate_shexp",{2048,1024,1,1}}, {"ffn_up_shexp",{2048,1024,1,1}},
            {"ffn_down_shexp",{1024,2048,1,1}}, {"ffn_gate_exps",{2048,512,256,1}},
            {"ffn_up_exps",{2048,512,256,1}}, {"ffn_down_exps",{512,2048,256,1}}});
        }
        for (const auto &[name,shape] : mats) {
            if(shape[2] == 1) load(name+".weight",shape,GGML_TYPE_NVFP4);
            load(name+".scale",{shape[2],1,1,1},GGML_TYPE_F32);
        }
        require(gguf_find_tensor(header,("blk."+std::to_string(block)+".ffn_exp_probs_b.bias").c_str()) < 0, "unexpected correction bias needs explicit handling");
    }
};

static Tensor *norm(Graph &g, Tensor *x, Tensor *gain, const std::string &name) {
    return g.mark(name,ggml_mul(g.ctx,ggml_rms_norm(g.ctx,x,1e-6f),gain));
}
static Tensor *mm(Graph &g, Weights &w, const std::string &stem, Tensor *x, const std::string &name) {
    auto *raw = g.mark(name+"_unscaled",ggml_mul_mat(g.ctx,w.at(stem+".weight"),x));
    ggml_prec_set_acc(raw,GGML_PREC_F32);
    return g.mark(name,ggml_mul(g.ctx,raw,w.at(stem+".scale")));
}
static Tensor *expert_mm(Graph &g, Weights &w, const std::string &stem, Tensor *x, Tensor *global_ids, Tensor *local_ids, int T, const std::string &name) {
    auto *raw = g.mark(name+"_unscaled",ggml_mul_mat_id(g.ctx,w.at(stem+".weight"),x,local_ids));
    ggml_prec_set_acc(raw,GGML_PREC_F32);
    auto *scales = ggml_reshape_3d(g.ctx,w.at(stem+".scale"),1,256,1);
    scales = ggml_repeat_4d(g.ctx,scales,1,256,T,1);
    scales = g.mark(name+"_selected_scale",ggml_get_rows(g.ctx,scales,global_ids));
    return g.mark(name,ggml_mul(g.ctx,raw,scales));
}
static Tensor *route(Graph &g, Tensor *logits, int T) {
    auto *probs = g.mark("router_probs",ggml_sigmoid(g.ctx,logits));
    auto *ids = g.mark("router_ids",ggml_argsort_top_k(g.ctx,probs,6));
    auto *selected = g.mark("router_selected_probs",ggml_get_rows(g.ctx,ggml_reshape_3d(g.ctx,probs,1,256,T),ids));
    selected = ggml_reshape_2d(g.ctx,selected,6,T);
    auto *sum = g.mark("router_sum",ggml_sum_rows(g.ctx,selected));
    auto *denom = g.mark("router_denominator",ggml_clamp(g.ctx,sum,6.103515625e-5f,INFINITY));
    g.mark("router_weights",ggml_reshape_3d(g.ctx,ggml_div(g.ctx,selected,denom),1,6,T));
    return ids;
}
static Tensor *named(Graph &g, const std::string &name) {
    for (auto &[s,t] : g.outputs) if (s==name) return t;
    throw std::runtime_error("missing graph name: "+name);
}
static void block(Weights &w, const fs::path &out, const std::vector<float> &values, const std::vector<int32_t> &positions) {
    const int T=positions.size();
    Graph g;
    require(T >= 1 && T <= 3 && values.size()==size_t(2048*T),"input shape differs");
    for(int t=0;t<T;++t) require(positions[t]>=0 && (t==0 || positions[t]>positions[t-1]),"invalid positions");
    for(float v:values) require(std::isfinite(v),"nonfinite residual");
    auto *x = g.input("input",2048,T,values);
    auto *pos = g.mark("positions",ggml_new_tensor_1d(g.ctx,GGML_TYPE_I32,T));
    memcpy(pos->data,positions.data(),T*4);
    auto *a = norm(g,x,w.at("attn_norm.weight"),"attn_norm");
    auto *q = ggml_reshape_3d(g.ctx,mm(g,w,"attn_q",a,"q_proj"),128,32,T);
    auto *k = ggml_reshape_3d(g.ctx,mm(g,w,"attn_k",a,"k_proj"),128,4,T);
    auto *v = ggml_reshape_3d(g.ctx,mm(g,w,"attn_v",a,"v_proj"),128,4,T);
    q = norm(g,q,w.at("attn_q_norm.weight"),"q_norm");
    k = norm(g,k,w.at("attn_k_norm.weight"),"k_norm");
    auto *qn=q; auto *kn=k;
    q = w.rope ? g.mark("q_rope",ggml_rope_ext(g.ctx,q,pos,nullptr,128,GGML_ROPE_TYPE_NEOX,262144,10000.0f,1.0f,0.0f,1.0f,0.0f,0.0f)) : g.mark("q_no_rope",q);
    k = w.rope ? g.mark("k_rope",ggml_rope_ext(g.ctx,k,pos,nullptr,128,GGML_ROPE_TYPE_NEOX,262144,10000.0f,1.0f,0.0f,1.0f,0.0f,0.0f)) : g.mark("k_no_rope",k);
    q = ggml_permute(g.ctx,q,0,2,1,3);
    k = ggml_permute(g.ctx,k,0,2,1,3);
    v = ggml_permute(g.ctx,v,0,2,1,3);
    auto *scores = g.mark("attention_scores",ggml_mul_mat(g.ctx,k,q));
    ggml_prec_set_acc(scores,GGML_PREC_F32);
    std::vector<float> mask(T*T);
    for (int i=0;i<T;++i) for (int j=0;j<T;++j) mask[i*T+j]=(positions[j]<=positions[i] && (!w.swa || positions[i]-positions[j]<513))?0.0f:-INFINITY;
    auto *mask_t=g.input("attention_mask",T,T,mask);
    auto *probs=ggml_soft_max_ext(g.ctx,scores,mask_t,1.0f/sqrtf(128.0f),0.0f);
    ggml_soft_max_add_sinks(probs,w.at("attn_sinks.weight"));
    g.mark("attention_probs",probs);
    v=ggml_cont(g.ctx,ggml_transpose(g.ctx,v));
    auto *heads=g.mark("attention_heads",ggml_mul_mat(g.ctx,v,probs));
    auto *joined=ggml_cont_2d(g.ctx,ggml_permute(g.ctx,heads,0,2,1,3),4096,T);
    auto *attn=mm(g,w,"attn_output",joined,"attention_output");
    auto *r=g.mark("ffn_inp",ggml_add(g.ctx,attn,x));
    auto *n=norm(g,r,w.at("ffn_norm.weight"),"ffn_norm");
    Tensor *logits=nullptr, *ids=nullptr, *ff=nullptr;
    std::vector<int32_t> selected_global;
    if(w.block==0) {
        auto *up=mm(g,w,"ffn_up",n,"dense_up");
        auto *gate=mm(g,w,"ffn_gate",n,"dense_gate");
        // Common loader initializes swiglu_clamp_shexp=0 before Flint's loader,
        // which does not override it (1ddffe9 llama-model.cpp:1471/1567).
        auto *act=g.mark("dense_swiglu",ggml_swiglu_split(g.ctx,gate,up));
        auto *di=g.mark("dense_down_input",act); // no MoE adaptive normalization
        auto *down=mm(g,w,"ffn_down",di,"dense_down");
        ff=g.mark("ffn_out",down);
    } else {
        logits=g.mark("router_logits",ggml_mul_mat(g.ctx,w.at("ffn_gate_inp.weight"),n));
        ggml_prec_set_acc(logits,GGML_PREC_F32);
        ids=route(g,logits,T);
        g.run(); // GGML computes routing before any selected expert payload read.
        selected_global=w.load_selected(ids,T);
        auto *local_ids=ggml_new_tensor_2d(g.ctx,GGML_TYPE_I32,6,T);
        memcpy(local_ids->data,w.local_selection.data(),6*T*4);
        auto *in3=ggml_reshape_3d(g.ctx,n,2048,1,T);
        auto *up=expert_mm(g,w,"ffn_up_exps",in3,ids,local_ids,T,"expert_up");
        auto *gate=expert_mm(g,w,"ffn_gate_exps",in3,ids,local_ids,T,"expert_gate");
        auto *act=g.mark("expert_swiglu",ggml_swiglu_split(g.ctx,gate,up));
        auto *amax=ggml_pool_1d(g.ctx,ggml_abs(g.ctx,act),GGML_OP_POOL_MAX,512,512,0);
        auto *e=ggml_scale(g.ctx,amax,1.0f/1024.0f);
        e=ggml_scale(g.ctx,ggml_log(g.ctx,e),1.4426950408889634f);
        e=ggml_clamp(g.ctx,ggml_ceil(g.ctx,e),0.0f,127.0f);
        e=g.mark("expert_down_scale",ggml_exp(g.ctx,ggml_scale(g.ctx,e,0.6931471805599453f)));
        auto *down_in=g.mark("expert_down_input",ggml_div(g.ctx,act,e));
        auto *down=expert_mm(g,w,"ffn_down_exps",down_in,ids,local_ids,T,"expert_down_before_rescale");
        down=g.mark("expert_down",ggml_mul(g.ctx,down,e));
        auto *weighted=g.mark("expert_weighted",ggml_mul(g.ctx,down,named(g,"router_weights")));
        auto *moe=ggml_view_2d(g.ctx,weighted,2048,T,weighted->nb[2],0);
        for (int slot=1;slot<6;++slot)
            moe=ggml_add(g.ctx,moe,ggml_view_2d(g.ctx,weighted,2048,T,weighted->nb[2],slot*weighted->nb[1]));
        g.mark("moe_out",moe);
        auto *su=mm(g,w,"ffn_up_shexp",n,"shared_up");
        auto *sg=mm(g,w,"ffn_gate_shexp",n,"shared_gate");
        auto *sa=g.mark("shared_swiglu",ggml_swiglu_split(g.ctx,sg,su));
        auto *sd=mm(g,w,"ffn_down_shexp",sa,"shared_down");
        ff=g.mark("ffn_out",ggml_add(g.ctx,moe,sd));
    }
    g.mark("l_out",ggml_add(g.ctx,ff,r));
    g.run(); g.dump(out);
    w.check_identity();
    if(ids) for(int t=0;t<T;++t) for(int slot=0;slot<6;++slot)
        require(static_cast<int32_t *>(ids->data)[t*ids->nb[1]/4+slot]==selected_global[t*6+slot],"GGML routing changed after sparse load");
    // Negative controls are attention-only GGML graphs, not new block or model runs.
    // They use actual full-attention Q/K/V and demonstrate gapped-fixture sensitivity.
    if((w.block==0 || w.block==5) && T==3) {
        require(positions==std::vector<int32_t>({0,512,513}),"full-attention T3 control requires gapped positions");
        auto changed = [](Tensor *a, Tensor *b) {
            require(ggml_nelements(a)==ggml_nelements(b),"control shape mismatch");
            size_t count=0;
            for(int64_t i=0;i<ggml_nelements(a);++i) {
                uint32_t x,y;
                memcpy(&x,static_cast<char *>(a->data)+i*4,4);
                memcpy(&y,static_cast<char *>(b->data)+i*4,4);
                count += x != y;
            }
            return count;
        };
        auto *wrong_q=ggml_rope_ext(g.ctx,qn,pos,nullptr,128,GGML_ROPE_TYPE_NEOX,262144,10000.0f,1.0f,0.0f,1.0f,0.0f,0.0f);
        auto *wrong_k=ggml_rope_ext(g.ctx,kn,pos,nullptr,128,GGML_ROPE_TYPE_NEOX,262144,10000.0f,1.0f,0.0f,1.0f,0.0f,0.0f);
        std::vector<float> wrong_mask(T*T);
        for(int i=0;i<T;++i) for(int j=0;j<T;++j)
            wrong_mask[i*T+j]=(positions[j]<=positions[i] && positions[i]-positions[j]<513)?0.0f:-INFINITY;
        auto *wrong_mt=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,T,T);
        memcpy(wrong_mt->data,wrong_mask.data(),wrong_mask.size()*4);
        auto *wrong_mp=ggml_soft_max_ext(g.ctx,scores,wrong_mt,1.0f/sqrtf(128.0f),0.0f);
        ggml_soft_max_add_sinks(wrong_mp,w.at("attn_sinks.weight"));
        auto *wrong_mh=ggml_mul_mat(g.ctx,v,wrong_mp);
        auto *wrong_rs=ggml_mul_mat(g.ctx,ggml_permute(g.ctx,wrong_k,0,2,1,3),ggml_permute(g.ctx,wrong_q,0,2,1,3));
        ggml_prec_set_acc(wrong_rs,GGML_PREC_F32);
        auto *wrong_rp=ggml_soft_max_ext(g.ctx,wrong_rs,mask_t,1.0f/sqrtf(128.0f),0.0f);
        ggml_soft_max_add_sinks(wrong_rp,w.at("attn_sinks.weight"));
        auto *wrong_rh=ggml_mul_mat(g.ctx,v,wrong_rp);
        auto *controls=ggml_new_graph_custom(g.ctx,4096,false);
        for(auto *t:{wrong_mp,wrong_mh,wrong_q,wrong_k,wrong_rs,wrong_rp,wrong_rh}) ggml_build_forward_expand(controls,t);
        require(ggml_graph_compute_with_ctx(g.ctx,controls,4)==GGML_STATUS_SUCCESS,"controls compute failed");
        std::ofstream control(out/"negative-controls.tsv");
        control << "deliberate_change\tstage\tbit_mismatches\n";
        for(const auto &[label,a,b]:std::vector<std::tuple<std::string,Tensor *,Tensor *>>{
            {"wrong_mask",mask_t,wrong_mt},{"wrong_mask_probs",probs,wrong_mp},{"wrong_mask_heads",heads,wrong_mh},
            {"wrong_rope_q",qn,wrong_q},{"wrong_rope_k",kn,wrong_k},{"wrong_rope_scores",scores,wrong_rs},
            {"wrong_rope_probs",probs,wrong_rp},{"wrong_rope_heads",heads,wrong_rh}}) {
            const auto n=changed(a,b); require(n>0,"negative control lacks sensitivity: "+label);
            control << label << "\tattention\t" << n << '\n';
        }
        require(static_cast<float *>(mask_t->data)[6]==0.0f && wrong_mask[6]==-INFINITY,"full/SWA distance513 control");
        control.close(); require(bool(control),"control receipt failed");
    }
    if(ids) {
    const auto *id_data=static_cast<int32_t *>(ids->data);
    std::ofstream margins(out/"router-margins.tsv");
    margins << "token\tselected_ids\tprob6_minus_prob7\tlogit6_minus_logit7\n";
    auto *p=static_cast<float *>(named(g,"router_probs")->data);
    auto *l=static_cast<float *>(logits->data);
    for(int t=0;t<T;++t) {
        std::vector<int> order(256); for(int e0=0;e0<256;++e0) order[e0]=e0;
        std::stable_sort(order.begin(),order.end(),[&](int i,int j){return p[t*256+i]>p[t*256+j];});
        margins << t << '\t';
        for(int j=0;j<6;++j) margins << (j?",":"") << id_data[t*ids->nb[1]/4+j];
        margins << '\t' << std::setprecision(17) << double(p[t*256+order[5]])-p[t*256+order[6]]
                << '\t' << double(l[t*256+order[5]])-l[t*256+order[6]] << '\n';
    }
    margins.close(); require(bool(margins),"margin receipt write failed");
    }
    if(w.block==0) {
        // Independent GGML controls of the actual dense activation; no fabricated
        // routed IDs/margins or compensated MoE normalization in this branch.
        auto *up=named(g,"dense_up"), *gate=named(g,"dense_gate");
        auto *sw=named(g,"dense_swiglu");
        auto *swapped=ggml_swiglu_split(g.ctx,up,gate);
        auto *wrong=ggml_swiglu_split(g.ctx,named(g,"dense_gate_unscaled"),up);
        wrong=ggml_mul(g.ctx,wrong,w.at("ffn_gate.scale"));
        auto *controls=ggml_new_graph_custom(g.ctx,4096,false);
        ggml_build_forward_expand(controls,swapped); ggml_build_forward_expand(controls,wrong);
        require(ggml_graph_compute_with_ctx(g.ctx,controls,4)==GGML_STATUS_SUCCESS,"dense controls failed");
        std::ofstream control(out/"dense-negative-controls.tsv");
        control << "deliberate_change\tbit_mismatches\n";
        for(const auto &[label,tensor]:std::vector<std::pair<std::string,Tensor *>>{
            {"swap_gate_up",swapped},{"gate_scale_after_silu",wrong}}) {
            size_t count=0;
            for(int64_t i=0;i<ggml_nelements(sw);++i) {
                uint32_t a,b; memcpy(&a,static_cast<char *>(sw->data)+i*4,4);
                memcpy(&b,static_cast<char *>(tensor->data)+i*4,4); count += a!=b;
            }
            require(count>0,"dense control lacks sensitivity: "+label);
            control << label << '\t' << count << '\n';
        }
        control.close(); require(bool(control),"dense control receipt failed");
        require(named(g,"dense_down_input")==sw,"dense down input must be unmodified activation");
    }
    std::cout << "block T=" << T << " outputs=" << g.outputs.size() << " status=success dir=" << out << std::endl;
}

template<typename T> static std::vector<T> read_input(const fs::path &path,size_t limit) {
    const auto size=fs::file_size(path);
    require(size>0 && size<=limit && size%sizeof(T)==0,"bounded input size differs");
    std::vector<T> values(size/sizeof(T));
    std::ifstream in(path,std::ios::binary);
    in.read(reinterpret_cast<char *>(values.data()),size);
    require(bool(in),"input read failed");
    return values;
}
int main(int argc,char **argv) {
    try {
        require(argc==6,"usage: flint-block-reference MODEL BLOCK INPUT_F32 POSITIONS_I32 NEW_OUTPUT");
        const fs::path out(argv[5]); require(!fs::exists(out),"output exists");
        const auto values=read_input<float>(argv[3],2048*3*4);
        const auto positions=read_input<int32_t>(argv[4],3*4);
        const fs::path receipts=out.string()+"-reads";
        require(fs::create_directory(receipts),"read receipt directory exists");
        std::cout << "tier=decoded_F32_math cpu_threads=4 flash=false KV=none activation_quantization=none\n";
        Weights w(argv[1],std::stoi(argv[2]),receipts/"tensor-ranges.tsv"); w.all();
        block(w,out,values,positions);
        std::cout << "reference_completed=true mary_parity_claim=false\n";
        return 0;
    } catch(const std::exception &e) { std::cerr << "ERROR: " << e.what() << std::endl; return 1; }
}
