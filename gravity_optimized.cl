// https://www.stat.uchicago.edu/~lekheng/courses/302/classics/greengard-rokhlin.pdf

#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

//typedef struct {double real; double imag;} cdouble_t;


// START USEFUL FUNCTIONS

bool in_rect(cdouble_t x, cdouble_t x1y1, cdouble_t x2y2) {
    // coords must be top left and bottom right corners
    return (x.real > x1y1.real && x.real < x2y2.real && x.imag < x1y1.imag && x.imag > x2y2.imag);
}

cdouble_t mesh_ij_to_box_center(int i, int j, int meshLength) {
    cdouble_t result;
    //double a = {{box_size}} / meshLength;
    result.real = ((i + 0.5) * {{box_size}} / meshLength) - {{box_size*0.5}};
    result.imag = ((j + 0.5) * {{box_size}} / meshLength) - {{box_size*0.5}};
    return result;
}
cdouble_t mesh_ij_to_nw_corner(int i, int j, int meshLength) {
    cdouble_t result;
    //double a = {{box_size}} / meshLength;
    result.real = (i * {{box_size}} / meshLength) - {{box_size*0.5}};
    result.imag = (j * {{box_size}} / meshLength) - {{box_size*0.5}};
    return result;
}
cdouble_t mesh_ij_to_se_corner(int i, int j, int meshLength) {
    cdouble_t result;
    //double a = {{box_size}} / meshLength;
    result.real = ((i+1) * {{box_size}} / meshLength) - {{box_size*0.5}};
    result.imag = ((j+1) * {{box_size}} / meshLength) - {{box_size*0.5}};
    return result;
}

void test_particles_in_meshbox(int meshi, int meshj, int meshLength, __global const cdouble_t *x, int* in_box_output){
    // if you need the count, please run this function and sum() it
    for (int i = 0; i < {{N_PARTICLES}}; i++){
        in_box_output[i] = in_rect(
            x[i],
            mesh_ij_to_nw_corner(meshi, meshj, meshLength),
            mesh_ij_to_se_corner(meshi, meshj, meshLength)
        );
    }
}
int sum(int* array, int N){
    int tot;
    for (int i=0; i<N; i++){
        tot += array[i];
    }
    return tot;
}


// END USEFUL FUNCTIONS

// STEP 1 OF GREENGARD
__kernel void calculate_multipoles(int meshLength, __global cdouble_t *mesh, __global const cdouble_t *x){
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    // first two indices selects our phi, mesh[i][j][*][*]
    long phi_i = meshLength * meshLength * meshLength * i + meshLength * meshLength * j;
    // a_k
    cdouble_t a[{{MULTIPOLE_TERMS}}] = {0};
    // buffer variables
    cdouble_t phi, z;
    
    // compute a_k for all the particles in the current box
    int in_box[{{N_PARTICLES}}] = {0};
    test_particles_in_meshbox(i, j, meshLength, x, in_box);
    // note that the current box is i,j in the first two AND the latter two coordinates
    // i.e. mesh[i][j][i][j]
    for (int k = 1; k < {{MULTIPOLE_TERMS+1}}; k++){
        for (int n = 0; n < {{N_PARTICLES}}; n++){
            if (in_box[n]) {
                a[k-1] = cdouble_add(a[k-1], cdouble_rmul(-{{m}} / k, x[n]));
            }
        }
    }

    // compute total charge
    double Q = {{m}}*sum(in_box, {{N_PARTICLES}});

    // compute phi(z) at each z=x+iy
    for (int x = 0; x < meshLength; x++){
        for (int y = 0; y < meshLength; y++){
            // see thm 2.1
            z = mesh_ij_to_box_center(x, y, meshLength);
            phi = cdouble_log(z);
            for (int k = 1; k < {{MULTIPOLE_TERMS+1}}; k++) {
                phi = cdouble_add(phi, cdouble_divide(a[k-1], cdouble_powr(z, k)));
            }
            mesh[phi_i + meshLength * x + y] = cdouble_rmul(Q, phi);
        }
    }
}

// STEP 2 OF GREENGARD
__kernel void calculate_coarse_multipole(int meshLength, __global cdouble_t *mesh, __global const cdouble_t *finest_mesh, __global const int *binomial_coeffs){
    
}



// STEP 6: direct eval in finest mesh
// STEP 7: add 

// add a way to pass all the in_box for the finest mesh, since we keep recalculating it