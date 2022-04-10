
#include <cuda_fp16.h>


__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
                            const mf_node *R,
                            long long nnz,
                            half *p,
                            half *q,
                            curandState *state,
                            float *dynamic_rate,
                            long long u_seg, //unused
                            long long v_seg,//unused
                            int k, //feature dimension vector 
                            int num_iters,
                            int current_iter,
                            int update_count_per_block, //unused
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_p,
                            float lambda_q,
                            double *gpu_iter_err,
                            int u_grid, //unused
                            int v_grid, //unused
                            int u_id, //unused
                            int v_id //unused
                            )
{

/*
In MF, one SGD update consists of four steps: 
1) read one sample (r[u,v] ) from the rating matrix, 
2) read two feature vectors (pu , qv ), 
3) compute prediction error(r[u,v] − pu * qv ), and 
4) update the features. Except for the first step, other three steps are all vector operations at length k.
*/
    //persistant thread
    for(int ite = current_iter; ite < current_iter + num_iters; ite ++){
        /*
        __ldg : Read-Only Data Cache Load Function
        T __ldg(const T* address);
        returns the data of type T located at address address,
        */
        float tmp_lrate = __ldg(&dynamic_rate[ite]); 
        for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++){

            int lane_id = threadIdx.x%32; //p_q_k_ind
            int local_wid = threadIdx.x/32; //
            int wid = 4*blockIdx.x + local_wid;  

            long long start_id = 0;
            
            /*only threads whose idx % 32 == 0 will access P and Q randomly  */
            /*
            if(lane_id == 0){ // 
                long long origin = (long long)(curand_uniform(&state[wid])*nnz);  //randum number gen. from uniform dist
                start_id = origin% nnz;
                //start_id == 0;
            }
            */

            /*
            __shfl is a  warp shuffling instruction
            note: __shfl is depreciated; use __shfl_sync instead
            shuffling is  used to compute the dot product pu × qv and broadcast the result. 
            usage: T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
                * __shfl_sync() returns the value of var held by the thread whose ID is given by srcLane. 
                * If width is less than warpSize then each subsection of the warp behaves as a separate entity 
                  with a starting logical lane ID of 0. 
                * If srcLane is outside the range [0:width-1], the value returned corresponds 
                  to the value of var held by the srcLane modulo width (i.e. within the same subsection).


            The __shfl_sync() intrinsics permit exchanging of a variable between threads within a warp 
            without use of shared memory. The exchange occurs simultaneously for all active threads within 
            the warp (and named in mask),  moving 4 or 8 bytes of data per thread depending on the type.
            Threads within a warp are referred to as lanes, and may have an index between 0 and warpSize-1 (inclusive). 
            Four source-lane addressing modes are supported: 
            */
            start_id = __shfl(start_id, 0); //Dr.Akoglu: 
            
            for(int i = 0;i < update_vector_size;i++){ 
                int offset = (start_id + i)%nnz;
                float r = __ldg( &R[offset].rate); //get the address of the rate field, read it from the cache
                int u = __ldg(&R[offset].u);
                int v = __ldg(&R[offset].v);

                //read the p & q into register file.
                //base_p and base_q are random, so access will not be coalseced
                //random b/c u,v are  fcts of offset, and offset is random if threadIdx.x % 32 = 0. 

                /*begin rahel's optimization*/
                int base_p  = u * k;
                int base_q = v* k;
                float tmp_u, tmp_v, tmp_prod;

                float tmp_p1 = __half2float(p[base_p + lane_id]);
                float tmp_q1 = __half2float(q[base_q + lane_id]);
    
                float tmp_p2 = __half2float(p[base_p + lane_id + 32]);
                float tmp_q2 = __half2float(q[base_q + lane_id + 32]);
            
                float tmp_p3 = __half2float(p[base_p + lane_id + 64]);
                float tmp_q3 = __half2float(q[base_q + lane_id + 64]);
            
                float tmp_p4 = __half2float(p[base_p + lane_id + 96]);
                float tmp_q4 = __half2float(q[base_q + lane_id + 96]);

                //get dot product.
                float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;
                tmp_product += __shfl_down(tmp_product, 16);
                tmp_product += __shfl_down(tmp_product, 8);
                tmp_product += __shfl_down(tmp_product, 4);
                tmp_product += __shfl_down(tmp_product, 2);
                tmp_product += __shfl_down(tmp_product, 1);
                tmp_product = __shfl(tmp_product,0);
                float ruv = r - tmp_product; //get error

                //update p and q
                /* end Rahel's opt*/
                //update p and q
                //only works for k=blockDim.x=128
                p[base_p + lane_id +  0] = __float2half(tmp_p1 + tmp_lrate*(ruv*tmp_q1 - lambda_p*tmp_p1));
                q[base_q + lane_id +  0] = __float2half(tmp_q1 + tmp_lrate*(ruv*tmp_p1 - lambda_q*tmp_q1));

                p[base_p + lane_id + 32] = __float2half(tmp_p2 + tmp_lrate*(ruv*tmp_q2 - lambda_p*tmp_p2));
                q[base_q + lane_id + 32] = __float2half(tmp_q2 + tmp_lrate*(ruv*tmp_p2 - lambda_q*tmp_q2));

                p[base_p + lane_id + 64] = __float2half(tmp_p3 + tmp_lrate*(ruv*tmp_q3 - lambda_p*tmp_p3));
                q[base_q + lane_id + 64] = __float2half(tmp_q3 + tmp_lrate*(ruv*tmp_p3 - lambda_q*tmp_q3));

                p[base_p + lane_id + 96] = __float2half(tmp_p4 + tmp_lrate*(ruv*tmp_q4 - lambda_p*tmp_p4));
                q[base_q + lane_id + 96] = __float2half(tmp_q4 + tmp_lrate*(ruv*tmp_p4 - lambda_q*tmp_q4));
            }  //end inside for: read p and q,
        }//end middle  for
    } //end outside for
}



