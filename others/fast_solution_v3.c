/*
'''
DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
Version 2, December 2004

Copyright(C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

0. You just DO WHAT THE FUCK YOU WANT TO.
'''

Python version written by tinrtgu
Rewritten in C by BytesInARow
Debugged and modified by anttip
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

//#include <inttypes.h>

#define featype long
//#define vtype float
#define vtype double
#define idxtype int

static char* train= "../data/train.csv";	// path to training file
static char* label= "../data/trainLabels.csv";  // path to label file of training data
static char* test= "../data/test.csv";  // path to testing file

int lineXreport= 100000;
int maxll;	// max line length
//idxtype De= 4; // 2^De = number of weights use for each model
idxtype De= 18; // 2^De = number of weights use for each model
//idxtype De= 24; // 2^De = number of weights use for each model
//idxtype De= 25; // 2^De = number of weights use for each model
vtype alpha= 0.1;   // learning rate for sgd optimization
//vtype alpha= 0.2;   // learning rate for sgd optimization

clock_t start, prev, now;

//static idxtype numthreads= 8;
static idxtype numthreads= 8;
static idxtype numfeat= 146;
static idxtype numlbl= 32;

idxtype lblXblk;

int pad64(int items, int sz) {return items + (64 - items*sz % 64) / sz;}

struct trainblk {
  idxtype blkidx, nlbl;
  vtype *y, *w, *n;
  idxtype padding[64-((2*sizeof(idxtype)+3*sizeof(vtype*)%64)/sizeof(idxtype))];
};

void build_trainblk(struct trainblk* _blk, idxtype _nlb, idxtype _D, idxtype _id) {
  _blk->blkidx= _id;
  _blk->nlbl= _nlb;
  _blk->y= (vtype *)calloc(sizeof(vtype), pad64(_nlb, sizeof(vtype)));
  _blk->w= (vtype *)calloc(sizeof(vtype), pad64(_nlb*_D, sizeof(vtype)));
  _blk->n= (vtype *)calloc(sizeof(vtype), pad64(_nlb*_D, sizeof(vtype)));
};

void free_trainblk(struct trainblk* _blk) {free(_blk->y); free(_blk->w); free(_blk->n);};

vtype logloss(vtype p, vtype y) {
  vtype q= fmax(fmin(p, (1.0 - 1e-15)), 1e-15);
  return y==1.0?-log(q):-log(1.0 - q);
}

vtype predict(featype *x, vtype *w, int xsz) {
  register int j;
  register vtype wTx= 0.0;
  register featype *x2= x;
  //for(j=0; j<numfeat; j++) wTx+= w[x[j]] * 1.0;  // w[i] * x[i], but if i in x we got x[i] = 1.
  for(j= 0; j++<numfeat;) wTx+= w[*x2++];
  return 1. / (1. + exp(-fmax(fmin(wTx, 20.0), -20.0)));  // bounded sigmoid
};

void update(idxtype lb, struct trainblk *tb, vtype p, vtype y, idxtype xsz, idxtype D, featype *features) {
  register int j;
  register featype i;
  register vtype delta= p - y;
  idxtype lb2= lb*D;
  for (j= 0; j<numfeat; j++) {
    //i = features[j];
    // alpha / sqrt(n) is the adaptive learning rate
    // (p - y) * x[i] is the current gradient
    // note that in our case, if i in x then x[i] = 1.
    //tb->n[lb*D+i]+= fabs(p - y);
    //tb->w[lb*D+i]-= (p - y) * 1. * alpha / sqrt(tb->n[lb*D+i]);
    i= features[j]+lb2;
    tb->n[i]+= fabs(delta);
    tb->w[i]-= delta * alpha / sqrt(tb->n[i]);
  }
};

static long hash(char *a) {
  register int len= strlen(a);
  register unsigned char *p= (unsigned char *)a;
  register long x= *p << 7;
  while (--len >= 0) x= (1000003 * x) ^ *p++; 
  x^= strlen(a);
  return x;
}

void split_feature_line(char *line, char delim, featype *features, char *idstr, idxtype D) {
  int feati= 1, ci= 0, lni= 0, wi= 0, jump= 1, i, j;
  char buffer[100];
  line[strlen(line)-1]= 0;
  while(jump) {
    while(line[lni]!=delim && line[lni]!=0) lni++;
    jump= line[lni] != 0 ? 1 : 0;
    line[lni]= 0;
    if (ci == 0) {
      strcpy(idstr, &line[wi]);
      ci++;
    } else {
      sprintf(buffer, "%i_%s", feati, &line[wi]);
      //features[feati++]= abs(hash(buffer))%D;
      features[feati++]= labs(hash(buffer))%D;
    }
    wi= lni+= jump;
  }
};

void split_label_line(char *line, char delim, struct trainblk **tb, int *skipcols) {
  int feati= 0, tbi= 0, fti= 0, lni= 0, jump= 0, skipc= 0;
  while(line[lni++]!=delim) continue;
  while(line[lni]!=0) {
    if (feati++==skipcols[skipc]) skipc++;
    else {
      line[lni+1]= 0;
      tb[tbi]->y[fti++]= atof(&line[lni]);
      if(fti>=tb[tbi]->nlbl) {
        tbi++;
        fti= 0;
      }
    }
    lni+=2;
  }
};

vtype fstep(struct trainblk *tb, idxtype numfeat, idxtype D, featype *features) {
  vtype err= 0.0;
  int l;
  for (l= 0; l < tb->nlbl; l++) {
    vtype p= predict(features, &tb->w[l*D], numfeat);
    update(l, tb, p, tb->y[l], numfeat, D, features);
    err+= logloss(p, tb->y[l]);  // for progressive validation
  }
  return err;
};

void ftrain(char *trainfile, char *labelfile, struct trainblk **tb, char delim, int *skiplblcols, int D) {
  vtype log_y14= log(1.0 - 1e-15);
  idxtype t, l= 0;
  featype *features= (featype *)calloc(sizeof(featype), pad64(numfeat, sizeof(featype)));
  char *ftline= (char *)malloc(maxll), *lbline= (char *)malloc(maxll), *idstr= (char *)malloc(maxll);
  FILE *ffeat= fopen(trainfile, "r"), *flbl= fopen(labelfile, "r");
  vtype err= 0.0, lnerr;
  if (ffeat == NULL || flbl == NULL) {perror("Error opening file"); return;}
  prev= start= clock();

  // skip headers
  fgets(ftline, maxll, ffeat); fgets(lbline, maxll, flbl);
  
  // per line loop
  while (fgets(ftline, maxll, ffeat) != NULL && fgets(lbline, maxll, flbl) != NULL) {
    split_feature_line(ftline, delim, features, idstr, D);
    split_label_line(lbline, delim, tb, skiplblcols);
    lnerr= 0.0;
#pragma omp parallel for reduction(+:lnerr) default (none) shared(tb, numfeat, numthreads, D, features)
    for (t = 0; t < numthreads; t++) lnerr+= fstep(tb[t], numfeat, D, features);
    err+= lnerr + log_y14;
    // print out progress, so that we know everything is working
    if (++l%lineXreport == 0) {
      now= clock();
      //printf("%d s\t%d ms\tencountered: %d\tcurrent logloss: %f\n", (now - start) / 1000, now - prev, l, (err / (float)numlbl) / l);
      printf("%d s\t%d ms\tencountered: %d\tcurrent logloss: %f\n", (now - start) / 1000, now - prev, l, (err / (numlbl+1)) / l);
      prev= now;
    }
  }
  fclose(flbl); fclose(ffeat);
};

void fpred(char *testfile, struct trainblk **tb, char delim, int D) {
  idxtype t, l, k, k2, ln= 0;
  featype *features= (featype *)calloc(sizeof(featype), pad64(numfeat, sizeof(featype)));
  char *ftline= (char *)malloc(maxll), *idstr= (char *)malloc(maxll);
  FILE *ffeat= fopen(testfile, "r"), *outfile= fopen("submission.csv", "w");
  prev= clock();

  // skip headers
  fgets(ftline, maxll, ffeat);

  fprintf(outfile, "id_label,pred\n");
  while (fgets(ftline, maxll, ffeat) != NULL) {
    split_feature_line(ftline, delim, features, idstr, D);
#pragma omp parallel for schedule(dynamic) default (none) shared(D, numfeat, numthreads, tb, features) private(l)
    for (t = 0; t<numthreads; t++) for (l = 0; l < tb[t]->nlbl; l++) tb[t]->y[l]= predict(features, &(tb[t]->w[l*D]), numfeat);

    //for (k= 0; k < numlbl; k++) {
    k2= 0;
    for (k= 0; k < numlbl+1; k++) {
      t= k2/lblXblk;
      //t= k/lblXblk;
      //l= k - t*lblXblk;
      //t= t >= numthreads ? numthreads - 1 : t;
      l= k2 - t*lblXblk;
      fprintf(outfile, "%s_y%d,%.11e\n", idstr, k + 1, k==13?0.0:tb[t]->y[l]);
      if (k!=13) k2++;
    }
    if (++ln%lineXreport == 0) {
      now= clock();
      printf("%d s\t%d ms\twritten: %d lines\n", (now-start) / 1000, now-prev, ln);
      prev= now;
    }
    
  }
  fclose(outfile); fclose(ffeat);
  free(features); free(ftline); free(idstr);
}

int main() {
  idxtype t, D= 1<<De, skiplblcols[3]= {13,-1};
  struct trainblk **tb;
    
  maxll= (numfeat + numlbl) * 64;
  lblXblk= numlbl/numthreads;
  
  tb= (struct trainblk **)calloc(sizeof(struct trainblk *), numthreads);
  for (t= 0; t<numthreads; t++) {
    tb[t]= (struct trainblk*)malloc(sizeof(struct trainblk));
    build_trainblk(tb[t], lblXblk, D, t);
  }

  ftrain(train, label, tb, ',', skiplblcols, D);
  fpred(test, tb, ',', D);
  
  for (t= 0; t < numthreads; t++) free_trainblk(tb[t]);
  free(tb);
  return 0;
}
