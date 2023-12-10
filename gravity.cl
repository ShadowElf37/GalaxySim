void step_position(private int gid, __global double *xx, __global double *xy, __global double *vx, __global double *vy){
    // initialize variables
    // int gid = get_global_id(0);
    xx[gid] += vx[gid]*{{TIME_STEP}};
    xy[gid] += vy[gid]*{{TIME_STEP}};
}

__kernel void step_gravity(__global double *xx, __global double *xy, __global double *vx, __global double *vy, __global double *a)
{
    // initialize variables
    int gid = get_global_id(0);
    double dvx = 0, dvy = 0;
    double dv_mag, r2, sr2, xdist, ydist;
    double p_mag;

    for (int i = 0; i < {{N_PARTICLES}}; i++){
        if (i == gid) continue;

        xdist = xx[gid]-xx[i];
        ydist = xy[gid]-xy[i];

        r2 = xdist*xdist + ydist*ydist;

        if (r2 > {{(0.3/box_size)**2}}) continue;

        sr2 = sqrt(r2);
        dv_mag = min({{TIME_STEP*G*m*m}}/r2 / sr2, {{CLAMP_VEL}});
        p_mag = min({{TIME_STEP}}/r2/r2, {{CLAMP_VEL*10000}});
        dvx -= (dv_mag - p_mag*{{PRESSURE_SCALE}}) * xdist;
        dvy -= (dv_mag - p_mag*{{PRESSURE_SCALE}}) * ydist;
    }

    //vx[gid] = clamp(vx[gid] + dvx, -{{CLAMP_VEL/1.414}}, {{CLAMP_VEL/1.414}});
    //vy[gid] = clamp(vy[gid] + dvy, -{{CLAMP_VEL/1.414}}, {{CLAMP_VEL/1.414}});
    vx[gid] = vx[gid] + dvx;
    vy[gid] = vy[gid] + dvy;
    a[gid] = sqrt(sqrt(dvx*dvx + dvy*dvy));

    step_position(gid, xx, xy, vx, vy);
}

uint coord_to_index (double x, double y){
    return convert_uint(round(x*{{IMG_X / (box_size)}}) + round(y*{{IMG_Y / (box_size)}}) * {{IMG_X}});
}

__kernel void to_image(
    __global const double *xx, __global const double *xy, __global const double *a, __global uchar *R, __global uchar *G, __global uchar *B)
{
    // initialize variables
    int gid = get_global_id(0);

    double x = xx[gid];
    double y = xy[gid];
    uint i;

    if (x < {{box_size}} && y > 0 && x > 0 && y < {{box_size}}){
        i = coord_to_index(x, y);
        R[i] = convert_uchar(255*a[gid]);
        G[i] = convert_uchar(255*fabs(0.3-a[gid]));
        B[i] = convert_uchar(255*max(0.5-a[gid], 0.));
    }
    
}

