#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

using namespace ACCCDSL;

void MM_AR_C() 
{
    Variable B(TensorElemType::Int32, "B");
    Variable S(TensorElemType::Int32, "S");    
    Variable H(TensorElemType::Int32, "H");

    Tensor w(Float16, {H,H}, Sliced(0), WORLD, RANK);
    Tensor b(Float16, H, Replicated, WORLD);
    Tensor in(Float16, {B,S,H}, Sliced(2), WORLD, RANK);
    Tensor r(Float16, {B,S,H}, Replicated, WORLD);

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = Update(w, w - lr * g1);
    
    Pipeline pipeline("model-parallel", {g,w,lr}, {w1});

    pipeline.codegen("mm-ar-c.cu");
}

void MM_RS_C_AG()
{
    Variable N(TensorElemType::Int32, "N");   
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");

    Stage g1 = ReduceScatter(Summation, g);
    Stage w1 = Scatter(w) - lr * g1;
    Stage w2 = Update(w, AllGather(w1));

    Pipeline pipeline("sgd", {g,w,lr}, {w2});
    
    pipeline.codegen("sgd-rs-c-ag.cu");
}

void ol_MM_fuse_RS_C_AG()
{
    Variable N(TensorElemType::Int32, "N");   
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");

    Stage g1 = ReduceScatter(Summation, g);
    Stage w1 = Scatter(w) - lr * g1;
    Stage w2 = Update(w, AllGather(w1));

    Pipeline pipeline("sgd", {g,w,lr}, {w2});
    pipeline.fuse({g1, w1, w2});
    pipeline.codegen("sgd-fuse-rs-c-ag.cu");
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./" << argv[0] << "<schedule>\n <schedule> is AR_C, RS_C_AG, fuse(RS_C_AG)\n" << std::endl;
        return 0;
    }
    
    std::string schedule = std::string(argv[1]);

    if (schedule == "MM_AR_C")
        MM_AR_C();
    else if (schedule == "MM_RS_C_AG")
        MM_RS_C_AG();
    else if (schedule == "OL_MM_fuse_RS_C_AG")
        ol_MM_fuse_RS_C_AG();
    else 
        std::cout << "Invalid schedule" << std::endl;

    return 0;
}