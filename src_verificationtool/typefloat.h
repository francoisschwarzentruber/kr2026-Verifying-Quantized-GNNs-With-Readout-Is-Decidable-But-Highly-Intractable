/**
 * classical float type (for comments see typecharsaturation.h)
 */

#define number float

void testNumber() {

}


float nondet_number() {
    return nondet_float();
}

float sumNumber(const float a, const float b) {
    return a + b;
}


float mulNumber(const float a, const float b) {
    return a * b;
}