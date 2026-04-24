#include <assert.h>

int n = 100;

typedef struct s_float
{
    int sign : 1;
    int exponent : 8;
    int mantissa : 23;
} my_float_struct;

typedef union float_dissector
{
    float f;
    unsigned b;
} float_dissector;

void checkForFloat()
{
    float f = nondet_float();
    float sumOf1 = 0;
    for (int i = 1; i < n; i++)
        sumOf1 += 1;

    float_dissector r1 = {.f = sumOf1 * f};

    float sumOfF = 0;
    for (int i = 1; i < n; i++)
        sumOfF += f;

    float_dissector r2 = {.f = sumOfF};

    assert(r1.b == r2.b);
}

char char_sum(char x, char y)
{
    return x + y;
}


char char_mul(char x, char y)
{
    return x * y;
}


void checkForChar()
{
    char f = nondet_char();
    char one = 3;

    float sumOf1 = 0;
    for (int i = 1; i < n; i++)
        sumOf1 = char_sum(sumOf1, one);

    char r1 = char_mul(sumOf1, f);

    char sumOfF = 0;
    for (int i = 1; i < n; i++)
        sumOfF = char_sum(sumOfF, char_mul(f, one));

    char r2 = sumOfF;

    assert(r1 == r2);
}

void main()
{
    // checkForFloat();
    checkForChar();
}