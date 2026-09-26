#include <routingkit/contraction_hierarchy.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using RoutingKit::ContractionHierarchy;
using RoutingKit::ContractionHierarchyQuery;

template<class T>
std::vector<T> read_raw(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("cannot open "+p);
    auto n=f.tellg(); if(n<0 || (uint64_t)n%sizeof(T)) throw std::runtime_error("bad size "+p);
    std::vector<T> v((size_t)n/sizeof(T)); f.seekg(0);
    if(!v.empty()) f.read((char*)v.data(),(std::streamsize)(v.size()*sizeof(T)));
    if(!f) throw std::runtime_error("short read "+p); return v;
}
template<class T>
void write_raw(const std::string&p,const std::vector<T>&v){
    std::ofstream f(p,std::ios::binary|std::ios::trunc);
    if(!f) throw std::runtime_error("cannot create "+p);
    if(!v.empty()) f.write((const char*)v.data(),(std::streamsize)(v.size()*sizeof(T)));
    if(!f) throw std::runtime_error("short write "+p);
}
int main(int argc,char**argv){
    try{
        if(argc<2) throw std::runtime_error("mode required");
        std::string mode=argv[1];
        if(mode=="build"){
            if(argc!=8) throw std::runtime_error("build node_count tail head weight ch threads");
            unsigned n=(unsigned)std::stoul(argv[2]);
            auto tail=read_raw<uint32_t>(argv[3]),head=read_raw<uint32_t>(argv[4]),w=read_raw<uint32_t>(argv[5]);
            if(tail.size()!=head.size()||tail.size()!=w.size()) throw std::runtime_error("arc length mismatch");
            std::cerr<<"QINGDAO_CH_BUILD nodes="<<n<<" arcs="<<tail.size()<<"\n";
            auto ch=ContractionHierarchy::build(n,
                std::vector<unsigned>(tail.begin(),tail.end()),
                std::vector<unsigned>(head.begin(),head.end()),
                std::vector<unsigned>(w.begin(),w.end()),
                [](std::string m){std::cerr<<"RK "<<m<<"\n";});
            ch.save_file(argv[6]);
            std::cerr<<"QINGDAO_CH_SAVED "<<argv[6]<<"\n";
            return 0;
        }
        if(mode=="query"){
            if(argc!=7) throw std::runtime_error("query ch qsrc qdst out threads");
            auto ch=ContractionHierarchy::load_file(argv[2]);
            auto s=read_raw<uint32_t>(argv[3]),t=read_raw<uint32_t>(argv[4]);
            if(s.size()!=t.size()) throw std::runtime_error("query length mismatch");
            int threads=std::max(1,std::stoi(argv[6]));
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            std::vector<uint32_t>d(s.size());
#pragma omp parallel
            {
                ContractionHierarchyQuery q(ch);
#pragma omp for schedule(dynamic,1024)
                for(long long i=0;i<(long long)s.size();++i){
                    if(s[(size_t)i]==t[(size_t)i]){d[(size_t)i]=0;continue;}
                    q.reset().add_source(s[(size_t)i]).add_target(t[(size_t)i]).run();
                    d[(size_t)i]=q.get_distance();
                }
            }
            write_raw(argv[5],d);
            auto mm=std::minmax_element(d.begin(),d.end());
            std::cerr<<"QINGDAO_CH_QUERY count="<<d.size()
                     <<" min="<<(d.empty()?0:*mm.first)<<" max="<<(d.empty()?0:*mm.second)<<"\n";
            return 0;
        }
        throw std::runtime_error("unknown mode "+mode);
    }catch(const std::exception&e){
        std::cerr<<"QINGDAO_CH_ERROR "<<e.what()<<"\n"; return 1;
    }
}
