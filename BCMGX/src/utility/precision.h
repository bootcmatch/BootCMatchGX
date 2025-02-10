#if defined(USE_SINGLE_PRECISION)
#define SQRT sqrtf
#define SIN sinf
#define ACOS acosf
#define EXP expf
#define FABS fabsf
#define REALafsai float
#define v(value) value##f
#else
#define SQRT sqrt
#define SIN sin
#define ACOS acos
#define EXP exp
#define FABS fabs
#define REALafsai double
#define v(value) value
#endif
