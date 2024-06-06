#pragma once

#define DETAILED_TIMING 1

extern float TOTAL_SOLRELAX_TIME;
extern float TOTAL_RESTGAMG_TIME;

extern float TOTAL_PRECAPPLY_TIME;
extern float TOTAL_SPMV_TIME;
extern float TOTAL_DOTP_TIME;
extern float TOTAL_AXPY_TIME;
extern float TOTAL_ALLREDUCE_TIME;
extern float TOTAL_SWORK_TIME;
extern float TOTAL_CUDAMEMCOPY_TIME;
extern float TOTAL_NORM_TIME;

extern float TOTAL_CSRVECTOR_TIME;
extern float TOTAL_TRIPLEPROD_TIME;
extern float TOTAL_DOUBLEMERGED_TIME;
extern float TOTAL_RESTPRE_TIME;

extern float SUITOR_TIME;
extern float TOTAL_MUL_TIME;
extern float TOTAL_MATCH_TIME;
extern float TOTAL_SETUP_TIME;
extern float TOTAL_MEM_TIME;
extern float TOTAL_RELAX_TIME;
extern float TOTAL_SHIFTED_CSRVEC;
extern float TOTAL_MAKE_P;
extern float TOTAL_TRA_P;
extern float TOTAL_MAKEAHW_TIME;
extern float TOTAL_MATCHINGPAIR_TIME;
extern float TOTAL_OTHER_TIME;
