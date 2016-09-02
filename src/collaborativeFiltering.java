import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import Jama.Matrix;
import java.util.HashSet;

public class collaborativeFiltering {
	public static double avg_mae=0.0;
	public static double avg_mse=0.0;
	public static void main(String[] args) throws NumberFormatException, IOException {
		String line;
		ArrayList<Integer> userIds = new ArrayList<Integer>();
		ArrayList<Integer> itemIds = new ArrayList<Integer>();
		ArrayList<Integer> ratings = new ArrayList<Integer>();
		ArrayList<Integer> test_userIds = new ArrayList<Integer>();
		ArrayList<Integer> test_itemIds = new ArrayList<Integer>();
		ArrayList<Integer> test_ratings = new ArrayList<Integer>();
		ArrayList<Integer> train_userIds = new ArrayList<Integer>();
		ArrayList<Integer> train_itemIds = new ArrayList<Integer>();
		ArrayList<Integer> train_ratings = new ArrayList<Integer>();
		
		BufferedReader br = new BufferedReader(new FileReader("input/u.data"));
		while ((line = br.readLine()) != null) {
			String rowData[] = line.split("\t");
			
			// To convert String data into integer
			userIds.add(Integer.parseInt(rowData[0]));
			itemIds.add(Integer.parseInt(rowData[1]));
			ratings.add(Integer.parseInt(rowData[2]));
		}
		br.close();
		
		ArrayList<Integer> uniqUserIds = new ArrayList<Integer>(new HashSet<Integer>(userIds));
		ArrayList<Integer> uniqItemIds = new ArrayList<Integer>(new HashSet<Integer>(itemIds));
				
		// storing data in Matrix
		Matrix R = new Matrix(uniqUserIds.size(), uniqItemIds.size());
		System.out.println("Start[Comment: The process might take a while around 1 min]");
		//5 fold cross-validation
		for(int k=0;k<5;k++){
			System.out.println("The calculated error for fold = "+ k);
			int start=(((userIds.size())/5)*k);
			int end;
			if(k!=4){
				end=(((userIds.size())/5)*k+1);
			}
			else{
				end =userIds.size();
			}
			
			for(int i=0; i<userIds.size();i++){
				if(i>=start && i< end){
					test_userIds.add(userIds.get(i));
					test_itemIds.add(itemIds.get(i));
					test_ratings.add(ratings.get(i));
				}
				else{
					train_userIds.add(userIds.get(i));
					train_itemIds.add(itemIds.get(i));
					train_ratings.add(ratings.get(i));
				}
				
			}
			
			//build rating matrix with train data
			for (int r = 0; r < train_userIds.size(); r++) {
		        R.set(train_userIds.get(r)-1, train_itemIds.get(r)-1, train_ratings.get(r));
			}
			
			// change vale of K for no.of features selection
			int K = 3;
			Matrix P = Matrix.random(uniqUserIds.size(), K);
			Matrix Q = Matrix.random(uniqItemIds.size(), K);
			
			Matrix nR = matrix_factorization(R, P, Q, K);
			calculate_error(R,nR,test_userIds,test_itemIds,test_ratings);
		}
		
		avg_mae=(avg_mae)/5;
		avg_mse=(avg_mse)/5;
		System.out.println("Average MAE: "+avg_mae);
		System.out.println("Average MSE: "+avg_mse);
		System.out.println("Done");
	}
	
	 private static void calculate_error(Matrix R, Matrix nR,ArrayList<Integer> userId, ArrayList<Integer> itemId,ArrayList<Integer> rating) {
		 // TODO Auto-generated method stub
		 double mae = 0;
		 double mse = 0;
		 int N = 0;
		 for(int i=0; i<userId.size();i++) {
			 double diff = nR.get(userId.get(i)-1, itemId.get(i)-1)-rating.get(i);
			 mae += Math.abs(diff);
			 mse += Math.pow(diff, 2);
			 N++;
		 }
		 mae = mae/N;
		 mse = Math.pow(mse/N,0.5);
		 avg_mae+=mae;
		 avg_mse+=mse;
		 System.out.println("MAE: "+mae);
		 System.out.println("MSE: "+mse);		 
	}

	public static Matrix matrix_factorization(Matrix R, Matrix P, Matrix Q, int K) {
		 Q = Q.transpose();
		 double alpha = 0.01;
		 double beta = 0.02;
		 for (int step=0; step<5000; step++) {
			 //System.out.println(step);
			 for (int i=0; i<R.getRowDimension(); i++) {
				 for (int j=0; j<R.getColumnDimension(); j++) {
					 if (R.get(i, j) > 0) {
						 double dotProduct = 0;
						 for (int idx=0; idx<P.getColumnDimension(); idx++) {
							 dotProduct += P.get(i, idx) * Q.get(idx, j);
						 }
						 double eij = R.get(i, j) - dotProduct;
						 for(int k=0; k<K; k++) {
							 double updateVal = P.get(i,k) + alpha * (eij * Q.get(k, j) - beta * P.get(i, k));
							 P.set(i, k, updateVal);
							 double updateVal2 = Q.get(k,j) + alpha * (eij * P.get(i,k) - beta * Q.get(k,j));
							 Q.set(k, j, updateVal2);
						 }
					 }
				 }
			 }
		 }
		 
		 return P.times(Q);
	 }

}
