public class linearRegression {
    public static void main(String[] args) {
        double l=0.01;
        double[] x={1,2,3,4};
        double[] y={3,5,6,7};
        double theta0= 0;
        double theta1=0;
        double sumh1=0;
        double sumh2=0;
        double sumh3=0;
        double j=0;
        double jtheta0=0;
        double jtheta1=0;
        for(int i=0; i<x.length; i++){
            double h0x= theta0 + theta1*x[i];
            double h1= h0x- y[i];
            double h2= h1*h1;
            double h3= h1*x[i];
            sumh1=sumh1+h1;
            sumh2=sumh2+h2;
            sumh3=sumh3+h3;
            j=sumh2/2*x.length;
            jtheta0=sumh1/x.length;
            jtheta1=sumh3/x.length;
        }
        theta0=theta0-l*jtheta0;
        theta1=theta1-l*jtheta1;
        System.out.println("summation of h0(x)-y=" + sumh1);
        System.out.println("summation of [h0(x)-y]^2=" + sumh2);
        System.out.println("summation of (h0(x)-y)x=" + sumh3);
        System.out.println("J=" + j);
        System.out.println("dj/dtheta0=" + jtheta0);
        System.out.println("dj/dtheta1=" + jtheta1);
        System.out.println("updated theta0= "+theta0);
        System.out.println("updated theta1= "+theta1);

    }
}
