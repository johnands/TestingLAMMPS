/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Yongnan Xiong (HNU), xyn@hnu.edu.cn
                         Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_nn_angular2.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include <fenv.h>

#include <fstream>
#include <iomanip>

using namespace LAMMPS_NS;
using std::cout;
using std::endl;

//#define MAXLINE 1024
//#define DELTA 4

/* ---------------------------------------------------------------------- */

PairNNAngular2::PairNNAngular2(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // We don't provide the force between two atoms only since it is Angular2
  restartinfo = 0;   // We don't write anything to restart file
  one_coeff = 1;     // only one coeff * * call
  manybody_flag = 1; // Not only a pair style since energies are computed from more than one neighbor
  cutoff = 10.0;      // Will be read from command line
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNAngular2::~PairNNAngular2()
{
  if (copymode) return;
  // If you allocate stuff you should delete and deallocate here. 
  // Allocation of normal vectors/matrices (not armadillo), should be created with
  // memory->create(...)

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

double PairNNAngular2::network(arma::mat inputVector) {
    // inputGraph vector is a 1xinputGraphs vector

    // linear activation for inputGraph layer
    m_preActivations[0] = inputVector;
    m_activations[0] = m_preActivations[0];

    // hidden layers
    for (int i=0; i < m_nLayers; i++) {
        // weights and biases starts at first hidden layer:
        // weights[0] are the weights connecting inputGraph layer to first hidden layer
        m_preActivations[i+1] = m_activations[i]*m_weights[i] + m_biases[i];
        m_activations[i+1] = sigmoid(m_preActivations[i+1]);
    }

    // linear activation for output layer
    m_preActivations[m_nLayers+1] = m_activations[m_nLayers]*m_weights[m_nLayers] + m_biases[m_nLayers];
    m_activations[m_nLayers+1] = m_preActivations[m_nLayers+1];

    // return activation of output neuron
    return m_activations[m_nLayers+1](0,0);
}

arma::mat PairNNAngular2::backPropagation() {
  // find derivate of output w.r.t. intput, i.e. dE/dr_ij
  // need to find the "error" terms for all the nodes in all the layers

  // the derivative of the output neuron's activation function w.r.t.
  // its inputGraph is propagated backwards.
  // the output activation function is f(x) = x, so this is 1
  arma::mat output(1,1); output.fill(1);
  m_derivatives[m_nLayers+1] = output;

  // we can thus compute the error vectors for the other layers
  for (int i=m_nLayers; i > 0; i--) {
      m_derivatives[i] = ( m_derivatives[i+1]*m_weightsTransposed[i] ) %
                         sigmoidDerivative(m_preActivations[i]);
  }

  // linear activation function for inputGraph neurons
  m_derivatives[0] = m_derivatives[1]*m_weightsTransposed[0];

  return m_derivatives[0];
}

arma::mat PairNNAngular2::sigmoid(arma::mat matrix) {

  return 1.0/(1 + arma::exp(-matrix));
}

arma::mat PairNNAngular2::sigmoidDerivative(arma::mat matrix) {

  arma::mat sigmoidMatrix = sigmoid(matrix);
  return sigmoidMatrix % (1 - sigmoidMatrix);
}

arma::mat PairNNAngular2::Fc(arma::mat R, double Rc, bool cut) {

  arma::mat value = 0.5*(arma::cos(m_pi*R/Rc) + 1);

  if (cut)
    for (int i=0; i < arma::size(R)(1); i++)
      if (R(0,i) > Rc) 
        value(0,i) = 0;
  
  return value;
}

double PairNNAngular2::Fc(double R, double Rc, bool cut) {

  if (cut)
    if (R < Rc)
      return 0.5*(cos(m_pi*R/Rc) + 1);
    else
      return 0;
  else
    return 0.5*(cos(m_pi*R/Rc) + 1);
}

arma::mat PairNNAngular2::dFcdR(arma::mat R, double Rc) {

  Rc = 1.0/Rc;
  return -(0.5*m_pi*Rc) * arma::sin(m_pi*R*Rc); 
}

double PairNNAngular2::dFcdR(double R, double Rc) {

  Rc = 1.0/Rc;
  return -(0.5*m_pi*Rc) * sin(m_pi*R*Rc);
}

double PairNNAngular2::G1(arma::mat Rij, double Rc) {

  return arma::accu( Fc(Rij, Rc, false) );
}

arma::mat PairNNAngular2::dG1dR(arma::mat Rij, double Rc) {

  return dFcdR(Rij, Rc);
}

double PairNNAngular2::G2(double rij, double eta, double Rc, double Rs) {

  return exp(-eta*(rij - Rs)*(rij - Rs)) * Fc(rij, Rc, false);
}

void PairNNAngular2::dG2dR(arma::mat Rij, double eta, double Rc, double Rs,
                          arma::mat &dG2) {

  dG2 = arma::exp(-eta*(Rij - Rs)%(Rij - Rs)) % 
          ( 2*eta*(Rs - Rij) % Fc(Rij, Rc, false) + dFcdR(Rij, Rc) );
}

double PairNNAngular2::G4(double rij, double rik, double rjk, double cosTheta, 
                          double eta, double Rc, double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         pow(1 + lambda*cosTheta, zeta) *  
         exp( -eta*(rij*rij + rik*rik + rjk*rjk) ) * 
         Fc(rij, Rc, false) * Fc(rik, Rc, false) * Fc(rjk, Rc, true);
}


void PairNNAngular2::dG4dR(double Rij, arma::mat Rik, arma::mat Rjk, 
                          arma::mat cosTheta, double eta, double Rc, 
                          double zeta, double lambda,
                          arma::mat &dEdR3, double *fj3,
                          arma::mat drij, arma::mat drik) {

  arma::mat powCosThetaM1 = pow(2, 1-zeta)*arma::pow(1 + lambda*cosTheta, zeta-1);
  arma::mat F1 = powCosThetaM1 % (1 + lambda*cosTheta);

  arma::mat F2 = arma::exp(-eta*(Rij*Rij + Rik%Rik + Rjk%Rjk));

  double FcRij = Fc(Rij, Rc, false);
  arma::mat FcRik = Fc(Rik, Rc, false);
  arma::mat FcRjk = Fc(Rjk, Rc, true);
  arma::mat F3 = FcRij * FcRik % FcRjk;

  arma::mat dF1dcosTheta = lambda*zeta*powCosThetaM1;
  arma::mat dF2dr = -2*eta*F2;
  arma::mat dF3drij = dFcdR(Rij, Rc) * FcRik % FcRjk;
  arma::mat dF3drik = FcRij * dFcdR(Rik, Rc) % FcRjk;

  arma::mat term1 = dF1dcosTheta % F2 % F3;
  arma::mat term2 = F1 % dF2dr % F3;
  arma::mat term3ij = F1 % F2 % dF3drij;
  arma::mat term3ik = F1 % F2 % dF3drik;

  double RijInv = 1.0 / Rij;
  arma::mat cosRijInv2 = cosTheta / (Rij*Rij);

  arma::mat RikInv = 1.0 / Rik;
  arma::mat cosRikInv2 = cosTheta / (Rik%Rik);

  arma::mat RijRikInv = 1.0 / (Rij*Rik);

  arma::mat termij = (cosRijInv2 % term1) - term2 - RijInv*term3ij;
  arma::mat termik = (cosRikInv2 % term1) - term2 - RikInv%term3ik;
  arma::mat crossTerm = -term1 % RijRikInv; 

  // all k's give a triplet energy contributon to atom j
  fj3[0] =  arma::accu( (drij(0,0) * termij) + (drik.row(0) % crossTerm) );
  fj3[1] =  arma::accu( (drij(1,0) * termij) + (drik.row(1) % crossTerm) );
  fj3[2] =  arma::accu( (drij(2,0) * termij) + (drik.row(2) % crossTerm) );

  // need all the different components of k's forces to do sum k > j
  dEdR3.row(0) =  (drik.row(0) % termik) + (drij(0,0) * crossTerm);
  dEdR3.row(1) =  (drik.row(1) % termik) + (drij(1,0) * crossTerm);
  dEdR3.row(2) =  (drik.row(2) % termik) + (drij(2,0) * crossTerm);
}

void PairNNAngular2::compute(int eflag, int vflag)
{
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  double evdwl = 0.0;
  eng_vdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    tagint itag = tag[i];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];

    // two-body interactions, skip half of them
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    int numshort = 0;

    // collect all neighbours in arma matrix, jnum max
    arma::mat Rij(1, jnum);        // all pairs (i,j)
    arma::mat drij(3, jnum);       // (dxij, dyij, dzyij)   
    std::vector<int> tagsj(jnum);  // indicies of j-atoms

    // store all triplets etc in vectors
    // jnum pairs, jnum-1 triplets max
    // for every (i,j) there is a vector of the below quantities
    std::vector<arma::mat> Riks(jnum-1);
    std::vector<arma::mat> driks(jnum-1);
    std::vector<arma::mat> cosThetas(jnum-1);
    std::vector<arma::mat> Rjks(jnum-1);
    std::vector<std::vector<int>> tagsk(jnum-1);

    // input vector to NN
    arma::mat inputVector(1, m_numberOfSymmFunc, arma::fill::zeros);

    // keep track of how many atoms below r2 and N3L
    int neighbours = 0; 

    // collect all pairs
    for (int jj = 0; jj < jnum; jj++) {

      int j = jlist[jj];
      j &= NEIGHMASK;
      tagint jtag = tag[j];

      double delxj = xtmp - x[j][0];
      double delyj = ytmp - x[j][1];
      double delzj = ztmp - x[j][2];
      double rsq1 = delxj*delxj + delyj*delyj + delzj*delzj;

      if (rsq1 >= cutoff*cutoff) continue;

      // store pair coordinates
      double rij = sqrt(rsq1);
      drij(0, neighbours) = delxj;
      drij(1, neighbours) = delyj;
      drij(2, neighbours) = delzj;
      Rij(0, neighbours) = rij;
      tagsj[neighbours] = j;

      // apply 2-body symmetry
      for (int s=0; s < m_numberOfSymmFunc; s++)
        if ( m_parameters[s].size() == 3 ) 
          inputVector(0,s) += G2(rij, m_parameters[s][0],
                                 m_parameters[s][1], m_parameters[s][2]);

      // collect triplets for this (i,j)
      arma::mat Rik(1, jnum-1);
      arma::mat drik(3, jnum-1);
      arma::mat CosTheta(1, jnum-1);
      arma::mat Rjk(1, jnum-1); 

      // three-body
      int neighk = 0;
      for (int kk = jj+1; kk < jnum; kk++) {

        int k = jlist[kk];
        k &= NEIGHMASK;

        double delxk = xtmp - x[k][0];
        double delyk = ytmp - x[k][1];
        double delzk = ztmp - x[k][2];
        double rsq2 = delxk*delxk + delyk*delyk + delzk*delzk;  

        if (rsq2 >= cutoff*cutoff) continue;
        
        // calculate quantites needed in G4
        double rik = sqrt(rsq2);
        double cosTheta = ( delxj*delxk + delyj*delyk + 
                            delzj*delzk ) / (rij*rik);
        double rjk = sqrt(rij*rij + rik*rik  - 2*rij*rik*cosTheta);

        // collect triplets
        drik(0, neighk) = delxk;
        drik(1, neighk) = delyk;
        drik(2, neighk) = delzk;
        Rik(0,neighk) = rik;
        CosTheta(0, neighk) = cosTheta;
        Rjk(0, neighk) = rjk;
        tagsk[neighbours].push_back(k);

        // increment
        neighk++;

        // apply 3-body symmetry
        for (int s=0; s < m_numberOfSymmFunc; s++) 
          if ( m_parameters[s].size() == 4 ) 
            inputVector(0,s) += G4(rij, rik, rjk, cosTheta,
                                   m_parameters[s][0], m_parameters[s][1], 
                                   m_parameters[s][2], m_parameters[s][3]);
      }

      // skip if no triplets left
      //if (neighk == 0) continue;

      // get rid of empty elements
      Rik = Rik.head_cols(neighk);
      drik = drik.head_cols(neighk);
      CosTheta = CosTheta.head_cols(neighk);
      Rjk = Rjk.head_cols(neighk);

      // store all k's for curren (i,j) to compute forces later
      Riks[neighbours]      = Rik;
      driks[neighbours]     = drik;
      cosThetas[neighbours] = CosTheta;
      Rjks[neighbours]      = Rjk;
      neighbours++;
    
      /*std::cout << "neighk " << neighk << std::endl;
      std::cout << "Rik: " << arma::size(Rik) << std::endl;
      std::cout << Rik << std::endl;
      std::cout << "drik: " << arma::size(drik) << std::endl;
      std::cout << drik << std::endl;
      std::cout << "cosTheta: " << arma::size(CosTheta) << std::endl;
      std::cout << CosTheta << std::endl;
      std::cout << "Rjk: " << arma::size(Rjk) << std::endl;
      std::cout << Rjk << std::endl;*/
    }

    // get rid of empty elements
    Rij = Rij.head_cols(neighbours);
    drij = drij.head_cols(neighbours);

    /*if (myStep == 19) {
      std::ofstream outfile;
      outfile.open("Tests/TestEnergy/inputVector2.txt");
      for (int p=0; p < arma::size(Rij)(1); p++){
        outfile << std::setprecision(14) << drij(0,p) << " " << 
        drij(1,p) << " " << drij(2,p) << " " <<
        Rij(0,p) << " ";
        cout << drij(0,p) << " " << drij(1,p) << " " << drij(2,p) << " " <<
        Rij(0,p) << " ";}
      outfile << endl;
      cout << endl;
      outfile.close();
    }*/

    /*std::cout << "neighbours " << neighbours << std::endl;
    std::cout << "Rij: " << arma::size(Rij) << std::endl;
    std::cout << Rij << std::endl;
    std::cout << "drij: " << arma::size(drij) << std::endl;
    std::cout << drij << std::endl;
    cout << "Riks0: " << Riks[0] << std::endl;
    cout << "Riks1: " << Riks[1] << std::endl;
    cout << "driks0: " << driks[0] << std::endl;
    cout << "driks1: " << driks[1] << std::endl;
    cout << "cosThetas0: " << cosThetas[0] << endl;
    cout << "cosThetas2: " << cosThetas[1] << endl;
    cout << "Rjks0: " << Rjks[0] << endl;
    cout << "Rjks1: " << Rjks[1] << endl;

    std::cout << "inputVector: " << inputVector << endl;*/

    // apply NN to get energy
    evdwl = network(inputVector);
    eng_vdwl += evdwl;

    // backpropagate to obtain gradient of NN
    arma::mat dEdG = backPropagation();
    
    // calculate forces by differentiating the symmetry functions
    // UNTRUE(?): dEdR(j) will be the force contribution from atom j on atom i
    for (int s=0; s < m_numberOfSymmFunc; s++) {
      
      // G2: one atomic pair environment per symmetry function
      if ( m_parameters[s].size() == 3 ) {

        arma::mat dG2(1,neighbours); // derivative of G2

        // calculate cerivative of G2 for all pairs
        // pass dG2 by reference instead of coyping matrices
        // and returning from function --> speed-up
        dG2dR(Rij, m_parameters[s][0],
              m_parameters[s][1], m_parameters[s][2], dG2);

        // chain rule. all pair foces
        arma::mat fpairs = -dEdG(0,s) * dG2 / Rij;

        // loop through all pairs for N3L
        for (int l=0; l < neighbours; l++) {
          double fpair = fpairs(0,l);
          f[i][0] += fpair*drij(0,l);
          f[i][1] += fpair*drij(1,l);
          f[i][2] += fpair*drij(2,l);

          // NOT N3L NOW
          f[tagsj[l]][0] -= fpair*drij(0,l);
          f[tagsj[l]][1] -= fpair*drij(1,l);
          f[tagsj[l]][2] -= fpair*drij(2,l);
        }
      }

      // G4/G5: neighbours-1 triplet environments per symmetry function
      else {
        
        for (int l=0; l < neighbours-1; l++) {

          int numberOfTriplets = arma::size(Riks[l])(1);

          double fj3[3];  // triplet force for atom j
          arma::mat dEdR3(3, numberOfTriplets); // triplet force for all atoms k

          // calculate forces for all triplets (i,j,k) for this (i,j)
          // fj3 and dEdR3 is passed by reference
          // all triplet forces are summed and stored for j in fj3
          // dEdR3 will contain triplet forces for all k, need 
          // each one seperately for N3L
          // Riks[l], Rjks[l], cosThetas[l] : (1,numberOfTriplets)
          // dEdR3, driks[l]: (3, numberOfTriplets)
          // drij.col(l): (1,3)
          dG4dR(Rij(0,l), Riks[l], Rjks[l], cosThetas[l],
                m_parameters[s][0], m_parameters[s][1], 
                m_parameters[s][2], m_parameters[s][3], 
                dEdR3, fj3, drij.col(l), driks[l]); 

          if (myStep == 19) {
            cout << "Rij: " << Rij(0,l) << endl;
            cout << "Rik: " << Riks[l] << endl;
            cout << "Rjk: " << Rjks[l] << endl;
            cout << "cosTheta: " << cosThetas[l] << endl;
            cout << "drij: " << drij.col(l) << endl;
            cout << "drik: " << driks[l] << endl;
            cout << m_parameters[s][0] << endl;
            cout << m_parameters[s][1] << endl;
            cout << m_parameters[s][2] << endl;
            cout << m_parameters[s][3] << endl;
            cout << fj3[0] << endl;
            cout << fj3[1] << endl;
            cout << fj3[2] << endl;
            cout << dEdR3 << endl;
            exit(1);
          }

          // N3L: add 3-body forces for i and k
          for (int m=0; m < numberOfTriplets; m++) {
            f[tagsk[l][m]][0] -= dEdG(0,s) * dEdR3(0,m);
            f[tagsk[l][m]][1] -= dEdG(0,s) * dEdR3(1,m);
            f[tagsk[l][m]][2] -= dEdG(0,s) * dEdR3(2,m);
          }

          // add 3-body forces for i, sum of j and k-forces
          f[i][0] += dEdG(0,s) * ( fj3[0] + arma::accu(dEdR3.row(0)) );
          f[i][1] += dEdG(0,s) * ( fj3[1] + arma::accu(dEdR3.row(1)) );
          f[i][2] += dEdG(0,s) * ( fj3[2] + arma::accu(dEdR3.row(2)) );

          // N3L: add 3-body force for j. fj3 is sum of all triplet forces
          f[tagsj[l]][0] -= dEdG(0,s) * fj3[0];
          f[tagsj[l]][1] -= dEdG(0,s) * fj3[1];
          f[tagsj[l]][2] -= dEdG(0,s) * fj3[2];
        }
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
  myStep++;
}

/* ---------------------------------------------------------------------- */

void PairNNAngular2::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq"); 
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNAngular2::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNAngular2::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 4)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read potential file and initialize potential parameters
  read_file(arg[2]);
  cutoff = force->numeric(FLERR,arg[3]);

  // let lammps know that we have set all parameters
  int n = atom->ntypes;
  int count = 0;
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
        setflag[i][j] = 1;
        count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNAngular2::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NN requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style NN requires newton pair on");

  // need a full neighbor list
  int irequest = neighbor->request(this);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNAngular2::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairNNAngular2::read_file(char *file)
{
  // convert to string 
  std::string trainingDir(file);
  std::string graphFile = trainingDir + "/graph.dat";
  std::cout << "Graph file: " << graphFile << std::endl;
  
  // open graph file
  std::ifstream inputGraph;
  inputGraph.open(graphFile.c_str(), std::ios::in);

  // check if file successfully opened
  if ( !inputGraph.is_open() ) std::cout << "File is not opened" << std::endl;

  // process first line
  std::string activation;
  inputGraph >> m_nLayers >> m_nNodes >> activation >> 
                m_numberOfInputs >> m_numberOfOutputs;
  std::cout << "Layers: "     << m_nLayers         << std::endl;
  std::cout << "Nodes: "      << m_nNodes          << std::endl;
  std::cout << "Activation: " << activation        << std::endl;
  std::cout << "Neighbours: " << m_numberOfInputs  << std::endl;
  std::cout << "Outputs: "    << m_numberOfOutputs << std::endl;

  // set sizes
  m_preActivations.resize(m_nLayers+2);
  m_activations.resize(m_nLayers+2);
  m_derivatives.resize(m_nLayers+2);

  // skip a blank line
  std::string dummyLine;
  std::getline(inputGraph, dummyLine);

  // process file
  // store all weights in a temporary vector
  // that will be reshaped later
  std::vector<arma::mat> weightsTemp;
  for ( std::string line; std::getline(inputGraph, line); ) {
    //std::cout << line << std::endl;

    if ( line.empty() )
        break;

    // store all weights in a vector
    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new weights on current line, add them to vector
    arma::mat matrix(1,m_nNodes);
    int i = 0;
    while ( ss >> buffer ) {
        matrix(0,i) = buffer;
        i++;
    }
    weightsTemp.push_back(matrix);
  }

  // can put all biases in vector directly
  // no need for temporary vector
  for ( std::string line; std::getline(inputGraph, line); ) {

    // store all weights in vector
    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new weights on current line, add them to vector
    arma::mat matrix(1,m_nNodes);
    int i = 0;
    while ( ss >> buffer ) {
        matrix(0,i) = buffer;
        i++;
    }
    m_biases.push_back(matrix);
  }

  // close file
  inputGraph.close();

  // write out all weights and biases
  /*for (const auto i : weightsTemp)
    std::cout << i << std::endl;
  std::cout << std::endl;
  for (const auto i : m_biases)
    std::cout << i << std::endl;*/

  // resize weights and biases matrices to correct shapes
  m_weights.resize(m_nLayers+1);

  // first hidden layer
  int currentRow = 0;
  m_weights[0]  = weightsTemp[currentRow];
  for (int i=0; i < m_numberOfInputs-1; i++) {
    currentRow++;
    m_weights[0] = arma::join_cols(m_weights[0], weightsTemp[currentRow]);
  }

  // following hidden layers
  for (int i=0; i < m_nLayers-1; i++) {
    currentRow++;
    m_weights[i+1] = weightsTemp[currentRow];
    for (int j=1; j < m_nNodes; j++) {
        currentRow++;
        m_weights[i+1] = arma::join_cols(m_weights[i+1], weightsTemp[currentRow]);
    }
  }

  // output layer
  currentRow++;
  arma::mat outputLayer = weightsTemp[currentRow];
  for (int i=0; i < m_numberOfOutputs-1; i++) {
    currentRow++;
    outputLayer = arma::join_cols(outputLayer, weightsTemp[currentRow]);
  }
  m_weights[m_nLayers] = arma::reshape(outputLayer, m_nNodes, m_numberOfOutputs);

  // reshape bias of output node
  m_biases[m_nLayers].shed_cols(1,m_nNodes-1);

  // obtained transposed matrices
  m_weightsTransposed.resize(m_nLayers+1);
  for (int i=0; i < m_weights.size(); i++)
    m_weightsTransposed[i] = m_weights[i].t();

  // write out entire system for comparison
  /*for (const auto i : m_weights)
    std::cout << i << std::endl;

  for (const auto i : m_biases)
    std::cout << i << std::endl;*/


  // read parameters file
  std::string parametersName = trainingDir + "/parameters.dat";

  std::cout << "Parameters file: " << parametersName << std::endl;

  std::ifstream inputParameters;
  inputParameters.open(parametersName.c_str(), std::ios::in);

  // check if file successfully opened
  if ( !inputParameters.is_open() ) std::cout << "File is not opened" << std::endl;

  inputParameters >> m_numberOfSymmFunc;

  std::cout << "Number of symmetry functions: " << m_numberOfSymmFunc << std::endl;

  // skip blank line
  std::getline(inputParameters, dummyLine);

  int i = 0;
  for ( std::string line; std::getline(inputParameters, line); ) {

    if ( line.empty() )
      break;

    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new parameters on current line, add them to matrix
    m_parameters.resize(m_numberOfSymmFunc);  
    while ( ss >> buffer ) {
        m_parameters[i].push_back(buffer);
    }
    i++;
  }
  inputParameters.close();
  std::cout << "File read......" << std::endl;
}
