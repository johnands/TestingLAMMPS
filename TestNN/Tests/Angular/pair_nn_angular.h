/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nn/angular,PairNNAngular)

#else

#ifndef LMP_PAIR_NN_ANGULAR_H
#define LMP_PAIR_NN_ANGULAR_H

#include "pair.h"

#include <armadillo>
#include <vector>

namespace LAMMPS_NS {

class PairNNAngular : public Pair {
 public:
  PairNNAngular(class LAMMPS *);
  virtual ~PairNNAngular();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  double network(arma::mat inputVector);
  arma::mat backPropagation();
  arma::mat sigmoid(arma::mat matrix);
  arma::mat sigmoidDerivative(arma::mat matrix);
  arma::mat Fc(arma::mat Rij, double Rc);
  double Fc(double Rij, double Rc);
  arma::mat dFcdR(arma::mat Rij, double Rc);
  double G1(arma::mat Rij, double Rc);
  arma::mat dG1dR(arma::mat Rij, double Rc);;
  double G2(arma::mat Rij, double eta, double Rc, double Rs);
  double G2(double Rij, double eta, double Rc, double Rs);
  arma::mat dG2dR(arma::mat Rij, double eta, double Rc, double Rs);
  double G4(arma::mat Rij, arma::mat Rik, arma::mat Rjk, arma::mat cosTheta, 
            double eta, double Rc, double zeta, double lambda);
  double G4(double Rij, arma::mat Rik, arma::mat Rjk, arma::mat cosTheta, 
            double eta, double Rc, double zeta, double lambda);
  double dG4dR(arma::mat Rij, arma::mat Rik, arma::mat Rjk, arma::mat cosTheta, 
               double eta, double Rc, double zeta, double lambda);

 protected:
  double cutoff;

  int m_nLayers;
  int m_nNodes;
  std::vector<arma::mat> m_weights            = std::vector<arma::mat>();
  std::vector<arma::mat> m_weightsTransposed  = std::vector<arma::mat>();
  std::vector<arma::mat> m_biases             = std::vector<arma::mat>();
  std::vector<arma::mat> m_preActivations     = std::vector<arma::mat>();
  std::vector<arma::mat> m_activations        = std::vector<arma::mat>();
  std::vector<arma::mat> m_derivatives        = std::vector<arma::mat>();
  std::vector<std::vector<double>> m_parameters;
  int m_numberOfInputs;
  int m_numberOfOutputs;
  int m_numberOfSymmFunc;
  int m_numberOfParameters;
  const double m_pi = arma::datum::pi;

  void allocate();
  void read_file(char *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style Vashishta requires atom IDs

This is a requirement to use the Vashishta potential.

E: Pair style Vashishta requires newton pair on

See the newton command.  This is a restriction to use the Vashishta
potential.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open Vashishta potential file %s

The specified Vashishta potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in Vashishta potential file

Incorrect number of words per line in the potential file.

E: Illegal Vashishta parameter

One or more of the coefficients defined in the potential file is
invalid.

E: Potential file has duplicate entry

The potential file has more than one entry for the same element.

E: Potential file is missing an entry

The potential file does not have a needed entry.

*/
