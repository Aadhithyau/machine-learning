import java.util.Arrays;
public class describe {
    public static void main(String[] args) {
        String[] name={"Tom","James","Ricky","Vin","Steve","Smith","Jack","Lee","David","Gasper","Betina","Andres"};
        int[] age={25,26,25,23,30,29,23,34,40,30,51,46};
        double[] rating={4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65};

        System.out.printf("%-10s %-15s %-5s%n", "NAME", "AGE", "RATING");
        System.out.printf("%-10s %-15s %-5s%n", "---", "----", "---");
        double asumfx=0;
        double rsumfx=0;

        for(int i=0; i<name.length; i++){
            System.out.printf("%-10s %-15d %-5.2f%n", name[i], age[i], rating[i]);
            asumfx=asumfx+age[i];
            rsumfx=rsumfx+rating[i];
        }

        System.out.println();
        System.out.printf("%-10s %-15s %-5s%n", "DESCRIBE", "AGE", "RATING");
        System.out.printf("%-10s %-15s %-5s%n", "---", "----", "---");
        mean(asumfx, rsumfx, name.length);
        minAndMax(age, rating);
        count(age.length, rating.length);
        std(age, rating, asumfx, rsumfx);
    }

    public static void mean(double a, double r, int l){
        double aResultMean=a/l;
        double rResultMean=r/l;
        finalResult("mean" ,aResultMean , rResultMean);
    }

    public static void count(double aCount, double rCount){
        finalResult("count", aCount, rCount);
    }

    public static void finalResult(String x, double aAnswer, double rAnswer){
        System.out.printf("%-10s %-15.2f %-5.2f%n", x, aAnswer, rAnswer);
    }

    public static void minAndMax(int[] aArray, double[] rArray){
        Arrays.sort(aArray);
        Arrays.sort(rArray);
        int maxAgeArray=aArray[aArray.length - 1]; 
        double maxRatingArray=rArray[rArray.length - 1]; 
        int minAgeArray=aArray[0];
        double minRatingArray=rArray[0];
        finalResult("min", minAgeArray, minRatingArray);
        finalResult("max", maxAgeArray, maxRatingArray);
    }
    public static void std (int[] ageArray, double[] ratingArray, double aSummation, double rSummation){
        aSummation=aSummation/ageArray.length;
        rSummation=rSummation/ratingArray.length;
        double aAnswer=0.0;
        double rAnswer=0.0;
        for(int i=0; i<ageArray.length; i++){
            aAnswer=aAnswer+(ageArray[i]-aSummation)*(ageArray[i]-aSummation);
            rAnswer=rAnswer+(ratingArray[i]-rSummation)*(ratingArray[i]-rSummation);
        }
        aAnswer=Math.sqrt(aAnswer/(ageArray.length-1));
        rAnswer=Math.sqrt(rAnswer/(ratingArray.length-1));
        finalResult("std", aAnswer, rAnswer);
    }
}
