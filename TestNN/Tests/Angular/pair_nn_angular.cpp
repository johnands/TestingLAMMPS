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
#include "pair_nn_angular.h"
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

#include <fstream>

using namespace LAMMPS_NS;
using std::cout;
using std::endl;

//#define MAXLINE 1024
//#define DELTA 4

/* ---------------------------------------------------------------------- */

PairNNAngular::PairNNAngular(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // We don't provide the force between two atoms only since it is Angular
  restartinfo = 0;   // We don't write anything to restart file
  one_coeff = 1;     // only one coeff * * call
  manybody_flag = 1; // Not only a pair style since energies are computed from more than one neighbor
  cutoff = 10.0;      // Will be read from command line
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNAngular::~PairNNAngular()
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

double PairNNAngular::network(arma::mat inputVector) {
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

arma::mat PairNNAngular::backPropagation() {
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

arma::mat PairNNAngular::sigmoid(arma::mat matrix) {

  return 1.0/(1 + arma::exp(-matrix));
}

arma::mat PairNNAngular::sigmoidDerivative(arma::mat matrix) {

  arma::mat sigmoidMatrix = sigmoid(matrix);
  return sigmoidMatrix % (1 - sigmoidMatrix);
}

arma::mat PairNNAngular::Fc(arma::mat Rij, double Rc) {

  return 0.5*(arma::cos(m_pi*Rij/Rc) + 1);
}

double PairNNAngular::Fc(double Rij, double Rc) {

  return 0.5*(cos(m_pi*Rij/Rc) + 1);
}

arma::mat PairNNAngular::dFcdR(arma::mat Rij, double Rc) {

  Rc = 1.0/Rc;
  return -(0.5*3.14*Rc) * arma::sin(m_pi*Rij*Rc); 
}

double PairNNAngular::G1(arma::mat Rij, double Rc) {

  return arma::accu( Fc(Rij, Rc) );
}

arma::mat PairNNAngular::dG1dR(arma::mat Rij, double Rc) {

  return dFcdR(Rij, Rc);
}

double PairNNAngular::G2(arma::mat Rij, double eta, double Rc, double Rs) {

  return arma::accu( arma::exp(-eta*(Rij - Rs)%(Rij - Rs)) % Fc(Rij, Rc) );
}

double PairNNAngular::G2(double Rij, double eta, double Rc, double Rs) {

  return exp(-eta*(Rij - Rs)*(Rij - Rs)) * Fc(Rij, Rc);
}

arma::mat PairNNAngular::dG2dR(arma::mat Rij, double eta, double Rc, double Rs) {

  return arma::exp(-eta*(Rij - Rs)%(Rij - Rs)) % 
         ( 2*eta*(Rs - Rij)%Fc(Rij, Rc) + dFcdR(Rij, Rc) );
}

double PairNNAngular::G4(arma::mat Rij, arma::mat Rik, arma::mat Rjk, 
                        arma::mat cosTheta, double eta, double Rc, 
                        double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         arma::accu( arma::pow(1 + lambda*cosTheta, zeta) % 
         arma::exp( -eta*(Rij%Rij + Rik%Rik + Rjk%Rjk) ) % 
         Fc(Rij, Rc) % Fc(Rik, Rc) % Fc(Rjk, Rc) );
}

double PairNNAngular::G4(double Rij, arma::mat Rik, arma::mat Rjk, 
                        arma::mat cosTheta, double eta, double Rc, 
                        double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         arma::accu( arma::pow(1 + lambda*cosTheta, zeta) % 
         arma::exp( -eta*(Rij*Rij + Rik%Rik + Rjk%Rjk) ) %
         (Fc(Rij, Rc) * Fc(Rik, Rc) % Fc(Rjk, Rc)) );
}


double PairNNAngular::dG4dR(arma::mat Rij, arma::mat Rik, arma::mat Rjk, 
                            arma::mat cosTheta, double eta, double Rc, 
                            double zeta, double lambda) {

  // to be implemented
  return 0;
}

void PairNNAngular::compute(int eflag, int vflag)
{

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

  std::ofstream outfile;
  outfile.open("testAngular.dat", std::ios::trunc);

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    tagint itag = tag[i];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    double fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    int numshort = 0;

    // collect all neighbours in arma matrix
    arma::mat Rij(1, 30);
    arma::mat Rik(1, 30);
    arma::mat drij(3, 30);
    arma::mat drik(3, 30);
    std::vector<int> tagsj(30);
    std::vector<int> tagsk(30);
    arma::mat inputVector(1, m_numberOfSymmFunc, arma::fill::zeros);
    bool write = 1;
    for (int jj = 0; jj < jnum; jj++) {
      int neighj = 0;
      int j = jlist[jj];
      j &= NEIGHMASK;
      tagint jtag = tag[j];

      /*if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }*/

      double delxj = xtmp - x[j][0];
      double delyj = ytmp - x[j][1];
      double delzj = ztmp - x[j][2];
      double rsq1 = delxj*delxj + delyj*delyj + delzj*delzj;

      if (rsq1 >= cutoff*cutoff) continue;

      // store coordinates of neighbour
      double rij = sqrt(rsq1);
      drij(0, neighj) = delxj;
      drij(1, neighj) = delyj;
      drij(2, neighj) = delzj;
      Rij(0, neighj) = rij;
      tagsj[neighj] = j;
      neighj++;


      // write position of neighbour j
      if (write)
      outfile << x[j][0] << " " << x[j][1] << " " << x[j][2] << " " << rsq1 <<
              " ";

      // three-body
      int neighk = 0;
      for (int kk = jj+1; kk < jnum; kk++) {

        int k = jlist[kk];
        k &= NEIGHMASK;

        double delxk = xtmp - x[k][0];
        double delyk = xtmp - x[k][1];
        double delzk = xtmp - x[k][2];
        double rsq2 = delxk*delxk + delyk*delyk + delzk*delzk;  

        if (rsq2 >= cutoff*cutoff) continue;

        // write position of neighbour k
        if (write)
        outfile << x[k][0] << " " << x[k][1] << " " << x[k][2] << " " << rsq2 <<
                " ";
        
        double rik = sqrt(rsq2);
        drik(0, neighk) = delxk;
        drik(1, neighk) = delyk;
        drik(2, neighk) = delzk;
        Rik(0,neighk) = rik;
        tagsk[neighk] = k;
        neighk++;
      }
      write = 0;

      // rij is now a single number, Rik is a matrix

      // get rid of empty elements
      Rij = Rij.head_cols(neighj);
      drij = drij.head_cols(neighj);
      Rik = Rik.head_cols(neighk);
      drik = drik.head_cols(neighk);

      // get theta and rjk
      // drij: (3,1)
      // rij: number, drik : (3,neighk)
      // Rik: (1,neighk)
      // drrik.row(0): (1,neighk) is all xik-coordinates and so on...
      // cosTheta: (1, neighk)
      arma::mat cosTheta = ( delxj*drik.row(0) + delyj*drik.row(1) + 
                             delzj*drik.row(2) ) / (rij*Rik);

      // Rik: (1,neighk), cosTheta: (1,neighk)
      // --> Rjk: (0,neighk)
      arma::mat Rjk = arma::sqrt(rij*rij + Rik%Rik  - 2*rij*Rik%cosTheta);

      /*std::cout << "neighj " << neighj << std::endl;
      std::cout << "neighk " << neighk << std::endl;
      std::cout << "Rij: " << arma::size(Rij) << std::endl;
      std::cout << Rij << std::endl;
      std::cout << rij << std::endl;
      std::cout << "drij: " << arma::size(drij) << std::endl;
      std::cout << drij << std::endl;
      std::cout << "Rik: " << arma::size(Rik) << std::endl;
      std::cout << Rik << std::endl;
      std::cout << "drik: " << arma::size(drik) << std::endl;
      std::cout << drik << std::endl;
      std::cout << "cosTheta: " << arma::size(cosTheta) << std::endl;
      std::cout << cosTheta << std::endl;
      std::cout << "Rjk: " << arma::size(Rjk) << std::endl;
      std::cout << Rjk << std::endl;*/

       
      // transform with symmetry functions, loop over all the parameters
      for (int s=0; s < m_numberOfSymmFunc; s++) {
        if ( m_parameters[s].size() == 3) 
          inputVector(0,s) += G2(rij, m_parameters[s][0],
                                 m_parameters[s][1], m_parameters[s][2]);
        else
          inputVector(0,s) += G4(rij, Rik, Rjk, cosTheta,
                                 m_parameters[s][0], m_parameters[s][1], 
                                 m_parameters[s][2], m_parameters[s][3]);
      }
    }
    outfile << endl;
    outfile << inputVector << std::endl;

    evdwl = network(inputVector);
    outfile << evdwl << endl;
    outfile.close();
    exit(1);

    /*std::ofstream outfile;
    outfile.open("exampleInputVector.dat", std::ios::trunc);

    for (int k=0; k < 70; k++) {
      outfile << inputVector(0,k) << std::endl;
    }
    outfile.close();*/

    // apply NN to get energy
    outfile << endl;
    outfile << inputVector << std::endl;

    evdwl = network(inputVector);
    outfile << evdwl << endl;
    outfile.close();
    exit(1);
    eng_vdwl += evdwl;

    // backpropagate to obtain gradient of NN
    arma::mat dEdG = backPropagation();
    
    // calculate forces by differentiating the symmetry functions
    // UNTRUE(?): dEdR(j) will be the force contribution from atom j on atom i
    /*arma::mat dEdR(1,neighbours, arma::fill::zeros);
    if (m_numberOfParameters == 1) 
      for (int s=0; s < m_numberOfSymmFunc; s++) 
        dEdR += dEdG(0,s) * dG1dR(Rij, m_parameters[s,0]);
    else
      for (int s=0; s < m_numberOfSymmFunc; s++)
        dEdR += dEdG(0,s) * dG2dR(Rij, m_parameters[s,0], 
                            m_parameters(s,1), m_parameters(s,2));

    // find total force
    for (int l=0; l < neighbours; l++) {
      double fpair = -dEdR(0,l) / Rij(0,l);
      f[i][0] += fpair*dr(l,0);
      f[i][1] += fpair*dr(l,1);
      f[i][2] += fpair*dr(l,2);
      f[tags[l]][0] -= fpair*dr(l,0);
      f[tags[l]][1] -= fpair*dr(l,1);
      f[tags[l]][2] -= fpair*dr(l,2);
    }

    f[i][0] -= fj[0] + fk[0];
    f[i][1] -= fj[1] + fk[1];
    f[i][2] -= fj[2] + fk[2];
    f[j][0] += fj[0];
    f[j][1] += fj[1];
    f[j][2] += fj[2];
    f[k][0] += fk[0];
    f[k][1] += fk[1];
    f[k][2] += fk[2];*/
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairNNAngular::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq"); 
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNAngular::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNAngular::coeff(int narg, char **arg)
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

void PairNNAngular::init_style()
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

double PairNNAngular::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairNNAngular::read_file(char *file)
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
