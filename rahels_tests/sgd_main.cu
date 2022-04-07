#include <"sgd_k128_kernel_hogwild_warp32_rahel.h">

int main(void){
    const mf_node *R;
    R.u = 0; R.v = 1; R.rate = 0.35;
    long long nnz = 1;
    half *p = 16;
    half *q = 16;
    curandState *state;
    float *dynamic_rate = 0.02;
    long long u_seg = 4; //unused
    long long v_seg = 3;//unused
    int k = 128; //feature dimension vector 
    int num_iters = 10;
    int current_iter = 0;
    int update_count_per_block = 3; //unused
    int update_count_this_block = 1;
    int update_vector_size = 100;
    float lambda_p = 0.01;
    float lambda_q = 0.02;
    double *gpu_iter_err;
        gpu_iter_err_val = 0.01;
    int u_grid; //unused
    int v_grid; //unused
    int u_id; //unused
    int v_id; //unused

   

    return 0;
}
