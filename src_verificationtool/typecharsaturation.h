/**
 * char with saturation meaning that 120+20 is 127 (CHAR_MAX)
 */


#define number char
#include <assert.h>
#include <limits.h>



/**
 * @returns a number (char here) chosen non-deterministically
 */
char nondet_number() {
    return nondet_char();
}

/**
 * two chars a and b
 * @returns the sum a+b but with saturation
 */
char sumNumber(const char a, const char b)
{
    int result = a + b;
    if (result > CHAR_MAX)
        return CHAR_MAX;
    if (result < CHAR_MIN)
        return CHAR_MIN;
    else
        return result;
}

/**
 * two chars a and b
 * @returns the sum a*b but with saturation
 */
char mulNumber(const char a, const char b)
{
    int result = a * b;
     if (result > CHAR_MAX)
        return CHAR_MAX;
    if (result < CHAR_MIN)
        return CHAR_MIN;
    else
        return result;
}


/**
 * unit tests to know whether the saturation works properly
 */
void testNumber()
{
    assert(sumNumber(-5, -120) == -125);
    assert(sumNumber(5, -120) == -115);
    assert(sumNumber(-8, -120) == CHAR_MIN);
    assert(sumNumber(120, 40) == CHAR_MAX);
}