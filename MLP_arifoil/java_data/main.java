import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class main {
	static String delimiter = "\t";
	static int inputDataLineNum = 160;
	static int alphaNum = 16;
	static int startAlpha = -5;
	static int endAlpha = 10;
	public static void main(String[] args){
		if (args.length != 1) {
			System.err.println("Input FolderName...");
			System.exit(1);  
		}

		try {
			int nacaNum=1; 	//Empty space of start
			String s1,s2,s3 = "";
			ArrayList<Data> dataList = new ArrayList();
			Data data;
			BufferedWriter[] data_alpha = new BufferedWriter[alphaNum];
			BufferedReader result = new BufferedReader(new FileReader(args[0] + "/result.txt"));

			result.readLine();
			s1 = result.readLine();
			for(int i=0;i<s1.indexOf('n');i++)
                s3 +=" ";
			//Check data List in Result.txt
			while(true){
				try{
					s2 = s1.split(s3)[nacaNum];
					data = new Data(s2);
					dataList.add(data);
					nacaNum++;
				}catch(ArrayIndexOutOfBoundsException e ){
					break;
				}
			}

			if (nacaNum < 1) {
				System.err.println("Result File Error...");
				System.exit(1);  
			}
			
			File folder=new File(args[0]);
			File []fileList=folder.listFiles();

			//Check data List in folder
			for(File tempFile : fileList) {
				if(tempFile.isFile()) {
					s1 = tempFile.getName();
					for (int i = 0; i < dataList.size(); i++) {
						if(dataList.get(i).check)
							continue;

						if(s1.indexOf(dataList.get(i).name)>-1){
							
							dataList.get(i).check = true;
							BufferedReader in = new BufferedReader(new FileReader(args[0] + "\\"+s1));
							String s;
							in.readLine();
							while ((s = in.readLine()) != null) {
								if(s.charAt(4)=='-')
									dataList.get(i).data_x += s.substring(4,13);
								else
									dataList.get(i).data_x += s.substring(5,13);
								dataList.get(i).data_x +=  delimiter;
								if(s.charAt(16)=='-')
									dataList.get(i).data_x += s.substring(16,25);
								else
									dataList.get(i).data_x += s.substring(17,25);
								dataList.get(i).data_x +=  delimiter;
								dataList.get(i).inputLineNum++;
							}
							in.close();
							
						}
					}

				}
			}

			
			//Check Read List
			int readCount = 0;
			for (int i = 0; i < dataList.size(); i++) {
				if(dataList.get(i).check){
					readCount++;
				}
			}
			
			if (readCount != dataList.size()) {
				System.err.println("Read Count : "+readCount);
				System.err.println("Real Count : "+dataList.size());
				System.err.println("Num of File not equal Num of data in second Line(result.txt)...");
				System.exit(1);  
			}
			
			int dataAt;
			result.readLine();
			s1 = result.readLine();
			while(s1 != null){
				for(int i=0;i<dataList.size();i++){
					try{
						
						dataAt =    (int) (Float.parseFloat(s1.split("    ")[i*2+1])+5);
						dataList.get(i).data_y[dataAt] = s1.split("    ")[i*2+2];
					}
					catch(NumberFormatException e){
						continue;
					}
					catch(ArrayIndexOutOfBoundsException e){
						System.err.println("result.txt has empty end line ...");
						System.exit(1);  
					}
				}
				
				s1 = result.readLine();	
			}
			
			String saveName,newLine;
			for(int i=0; i<alphaNum; i++){
				saveName = args[0] + "/data_Alpha_" + (i+startAlpha)+".txt";
				data_alpha[i] = new BufferedWriter(new FileWriter(saveName, true));
				for(int j=0;j<dataList.size();j++){
					if(dataList.get(j).data_y[i] !=null){
						newLine = dataList.get(j).name + delimiter + dataList.get(j).data_x + dataList.get(j).data_y[i];
//						System.out.println(newLine);
						data_alpha[i].write(newLine);
						data_alpha[i].newLine();
					}
				}
				data_alpha[i].close();
				System.out.println(saveName + " End...");
			}
			
			

		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

	static class Data{
		String name;
		String data_x;
		String[] data_y;
		int inputLineNum;
		boolean check = false;
		Data(){}
		Data(String name){
			data_x="";
			data_y = new String[alphaNum];
			inputLineNum=0;
			this.name = name;
		}
	}
}
