/* header file for the main MCMC loop */

#ifndef DRIVE_H
#define DRIVE_H


#include <cstdio>

typedef struct HyperparData
{
    // members
    const double mrfA;
    const double mrfB;
    const unsigned int *mrfG;
    const double *mrfG_weights;
    const int mrfG_edge_n;
    const double piA;
    const double piB;

    const double mrfA_prop;
    const double mrfB_prop;
    const unsigned int *mrfG_prop;
    const double *mrfG_prop_weights;
    const int mrfG_prop_edge_n;
    const double rhoA;
    const double rhoB;

    // const double vSq;
    const double vA;
    const double vB;
    // const double v0Sq;
    const double v0A;
    const double v0B;
    // const double tau0Sq;
    const double tau0A;
    const double tau0B;
    // const double tauSq;
    const double tauA;
    const double tauB;
    // const double wSq;
    const double wA;
    const double wB;
    // const double w0Sq;
    const double w0A;
    const double w0B;
    const bool w0IGamma;

    const bool kappaIGamma;
    const double kappaA;
    const double kappaB;
} hyperparS;

/*
typedef struct key_parameters
{
    // members
    double *xi;
    double *zetas;
    double *betas;
    double *kappa;
} keyParS;
*/



#endif
