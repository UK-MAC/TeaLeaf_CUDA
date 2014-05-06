/*
** The function
**                 tqli()
** determine eigenvalues and eigenvectors of a real symmetric
** tri-diagonal matrix, or a real, symmetric matrix previously
** reduced by function tred2[] to tri-diagonal form. On input,
** d[] contains the diagonal element and e[] the sub-diagonal
** of the tri-diagonal matrix. On output d[] contains the
** eigenvalues and  e[] is destroyed. If eigenvectors are
** desired z[][] on input contains the identity matrix. If
** eigenvectors of a matrix reduced by tred2() are required,
** then z[][] on input is the matrix output from tred2().
** On output, the k'th column returns the normalized eigenvector
** corresponding to d[k]. 
** The function is modified from the version in Numerical recipe.
*/

#include <math.h>
#define   SIGN(a,b) ((b)<0 ? -fabs(a) : fabs(a))

double pythag(double a, double b)
{
  double absa,absb;
  absa=fabs(a);
  absb=fabs(b);
  if (absa > absb) return
absa*sqrt(1.0+(absb/absa)*(absb/absa));
  else return 
(absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}
// End: function pythag(), (C) Copr. 1986-92 Numerical Recipes Software )%.


void tqli_(double *d, double *e, int *np, double *z, int* info)
{    

   const int n = *np;
   int   m,l,iter,i,k;
   double         s,r,p,g,f,dd,c,b;
   *info = 0;

   for(i = 1; i < n; i++) e[i-1] = e[i];
   e[n] = 0.0;
   for(l = 0; l < n; l++) {
      iter = 0;
      do {
         for(m = l; m < n-1; m++) {
            dd = fabs(d[m]) + fabs(d[m+1]);
            if((double)(fabs(e[m])+dd) == dd) break;
         }
         if(m != l) {
            if(iter++ == 30) {
               *info = 1;
               return;
            }
            g = (d[l+1] - d[l])/(2.0 * e[l]);
            r = pythag(g,1.0);
            g = d[m]-d[l]+e[l]/(g+SIGN(r,g));
            s = c = 1.0;
            p = 0.0;
            for(i = m-1; i >= l; i--) {
               f      = s * e[i];
               b      = c*e[i];
               e[i+1] = (r=pythag(f,g));
               if(r == 0.0) {
                  d[i+1] -= p;
                  e[m]    = 0.0;
                  break;
               }
               s      = f/r;
               c      = g/r;
               g      = d[i+1] - p;
               r      = (d[i] - g) * s + 2.0 * c * b;
               d[i+1] = g + (p = s * r);
               g      = c * r - b;
               /*
               for(k = 0; k < n; k++) {
                  f         = z[k][i+1];
                  z[k][i+1] = s * z[k][i] + c * f;
                  z[k][i]   = c * z[k][i] - s * f;
               }
               */
            } /* end i-loop */
            if(r == 0.0 && i >= l) continue;
            d[l] -= p;
            e[l]  = g;
            e[m]  = 0.0;
         } /* end if-loop for m != 1 */
      } while(m != l);
   } 
/* end l-loop */
} /* End: function tqli(), (C) Copr. 1986-92 Numerical Recipes Software )%. */

