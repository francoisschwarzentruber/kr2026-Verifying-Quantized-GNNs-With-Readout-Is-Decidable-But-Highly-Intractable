/**
 * This is the header for veryfing GNNs. It supposes a type "number" with functions addNumber, mulNumber that are respectively addition and multiplication for that
 * number type.
 */

#include <assert.h>
#include <stdbool.h>


/**
 *  declare a feature. A feature is a value appearing at each vertex of the graph. It can be an input feature or an intermediate value
 */
#define feature(name) number name[Nbound] = {0};

// adjacency matrix of the input graph
bool E[Nbound][Nbound];

/**
 * @param f a feature
 * @description declare the feature as unknown
 */
void unknownFeature(number f[])
{
  for (unsigned int i = 0; i < N; i++)
    f[i] = nondet_number();
}

/**
 * @description declare the input graph as unknown (the adjacency matrix is unknown)
 * It means that we non-deterministically choose a matrix E
 */
void unknownGraph()
{
  for (unsigned int i = 0; i < N; i++)
    for (unsigned int j = 0; j < N; j++)
      E[i][j] = nondet_bool();
}

/**
 *  @description take feature r and add v to it
 */
void add(number r[], const number v[])
{
  for (unsigned int i = 0; i < N; i++)
    r[i] = sumNumber(r[i], v[i]);
}

/**
 *  @description take feature r and add the cte to it
 */
void addCte(number r[], const number cte)
{
  for (int i = 0; i < N; i++)
    r[i] = sumNumber(r[i], cte);
}

/**
 *  @description puts scalar * v in r
 */
void mul(number r[], number scalar, number v[])
{
  for (int i = 0; i < N; i++)
    r[i] = sumNumber(r[i], mulNumber(scalar, v[i]));
}

/**
 *  @description compute ReLU to feature u and puts the result in r
 */
void ReLU(number r[], const number u[])
{
  for (int i = 0; i < N; i++)
    r[i] = u[i] >= 0 ? u[i] : 0;
}

void ReLUp(number r[], const number u[], number p)
{
  for (int i = 0; i < N; i++) {
    if (u[i] < 0) {
      r[i] = 0;
    } else if (u[i] > p) {
      r[i] = p;
    } else {
      r[i] = u[i];
    }
  }
}

void trReLU(number r[], const number u[])
{
  for (int i = 0; i < N; i++) {
    if (u[i] < 0) {
      r[i] = 0;
    } else if (u[i] > 1) {
      r[i] = 1;
    } else {
      r[i] = u[i];
    }
  }
}


/**
 * @description compute the aggregation (sum) of feature f on the neighbor and aggregate it in r
 */
void agg(number r[], const number f[])
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (E[i][j])
        r[i] = sumNumber(r[i], f[j]);
}

/**
 * @description compute the aggregation (sum) of feature f everywhere and aggregate it in r
 */
void aggG(number r[], const number f[])
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      r[i] = sumNumber(r[i], f[j]);
}
