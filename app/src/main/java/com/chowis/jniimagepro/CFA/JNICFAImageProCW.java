package com.chowis.jniimagepro.CFA;

public class JNICFAImageProCW
{

	static
	{
		System.loadLibrary("JNICFAImageProCWCore");
	}

	public native String getVersionJni();
	public native String getMakeDateJni();
	//Utils
	public native double get3DTextureJni(String sInputPath, String sOutputPath);
	public native double FFACropWideJni(String sInputPath, String sOutputPath);
	public native double skinGroupFFAJni(String sInputPath, String sOutputPath);
	
	// CFA Masking
	// 2022-03-26, updated by Shu Li
	public native double readMaskCFAWrinklesJni(String sInputPath, String sPathM, String sOutputPath);
	public native double readMaskCFAPoresJni(String sInputPath, String sPathM, String sOutputPath);
	public native double readMaskCFAImpuritiesJni(String sInputPath, String sPathM, String sOutputPath);
	public native double readMaskCFARednessJni(String sInputPath, String sPathM, String sOutputPath);
	public native double readMaskCFASpotsJni(String sInputPath, String sPathM, String sOutputPath);
	public native double readMaskCFADarkCircleJni(String sInputPath, String sPathM, String sOutputPath);
	public native double readMaskCFARadianceJni(String sInputPath, String sPathW, String sPathG, String sOutputPath);
	public native double readMaskCFAOilinessJni(String sInputPath, String sPathW, String sPathG, String sOutputPath);
	// 2022-03-26, added by Shu Li
    public native double readMaskCFAHyperPigmentationJni(String sInputPath, String sPathM, String sOutputPath);
    
    // 2022-11-29, updated by Shu Li.
    public native double checkImgQualityJni(String sInputPath);
    
    // CFA Local algorithms.
    // 2023-03-07, updated by Shu Li.
    public native String CFALocalRedness104Jni(
    		String xplFrontImgPath,
    		String xplLeftImgPath,
    		String xplRightImgPath,
    		String rednessFrontRoiImgPath,
    		String rednessLeftRoiImgPath,
    		String rednessRightRoiImgPath,
    		String frontResultOutputPath,
    		String leftResultOutputPath,
    		String rightResultOutputPath,
    		String frontMaskOutputPath,
    		String leftMaskOutputPath,
    		String rightMaskOutputPath,
    		double usedFrontCamera,
    		double sideImageEnabled);
    
    public native String CFALocalOiliness100Jni(
    		String pplFrontOriginalImgPath,
    		String oilinessRoiImgPath,
    		String resultImgOutputPath,
    		String greenMaskImgOutputPath,
    		String whiteMaskImgOutputPath,
    		double usedFrontCamera);
    
    public native String CFALocalRadianceDullness100Jni(
    		String pplFrontOriginalImgPath, 
    		String radianceDullnessFrontRoiImgPath, 
    		String resultImgOutputPath, 
    		String grayMaskImgOutputPath, 
			String whiteMaskImgOutputPath,
			double usedFrontCamera);
    
    public native String CFALocalImpurities104Jni(
    		String uvlFrontOriginalImgPath,
    		String impuritiesFrontRoiImgPath,
    		String resultImgOutputPath, 
    		String maskImgOutputPath,
    		double usedFrontCamera);
    
    public native String CFALocalPores100Jni(
    		String pplFrontOriginalImgPath,
    		String inputRoiImgPath,
    		String resultImgOutputPath, 
    		String maskImgOutputPath,
    		double usedFrontCamera);
	
    public native double CFALocalElasticity100Jni(double wrinkleScore, double dullnessScore);
    
    public native double CFAGetAnalyzedImgJni(String inputOriginalImgPath, String inputMaskImgPath, String outputAnalyzedImgPath, double alpha, int maskB, int maskG, int maskR, int contourB, int contourG, int contourR);
}
