mf_model*sgd_train(mf_problem*tr, mf_problem*te, Parameter para)
{

    clock_t start;

    //collect the factor. scaling is used to make sure every rating is around 1.
    SGDRate ave;
    SGDRate std_dev;
    SGDRate scale = 1.0;

    //preprocess stuff
    collect_data(tr, ave, std_dev);
    scale = max((SGDRate)1e-4, std_dev);

    fflush(stdout);

    //shuffle the u & v randomly to: 1) increase randomness. 2) block balance.
    printf("shuffle problem ...\n");
    start = clock();
    int* p_map = gen_random_map(tr->m);
    int* q_map = gen_random_map(tr->n);
    int* inv_p_map = gen_inv_map(p_map, tr->m);
    int* inv_q_map = gen_inv_map(q_map, tr->n);

    shuffle_problem(tr, p_map, q_map);

    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");

    //problem grid.
    grid_problem(tr); 

    //scale problem
    printf("scale problem ...\n");
    start = clock();
    scale_problem(tr, 1.0/scale, tr->u_seg, tr->v_seg);
    para.lambda_p = para.lambda_p/scale;
    para.lambda_q = para.lambda_q/scale;
    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");

    //init model
    mf_model*model = init_model(tr, para.k, ave/std_dev);

    //train
    sgd_update_k128(para, model, tr, scale);

    //scale model
    scale_model(model, scale);

    //shuffle model
    shuffle_model(model, inv_p_map, inv_q_map);
    return model;