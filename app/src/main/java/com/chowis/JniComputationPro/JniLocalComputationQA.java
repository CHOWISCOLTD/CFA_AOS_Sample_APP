package com.chowis.JniComputationPro;

public class JniLocalComputationQA
{
	static 
	{
		System.loadLibrary("JniLocalComputationQA");
	}
	
	public native String getVersionJni();
	public native String getMakeDateJni();
	
	////////////////////////////////////2023-03-21 UPDATED BY SHU LI ////////////////////////////////////////////
	//
	//
	// ----------- (1) ----------- Computation for all skin projects, including CNDP Skin, CFA, FFA, and CMA2-Skin.
    public native double computeSkinAge101Jni(double wrinkleScore, double pigmentationSpotsScore, double realBiologicalAge);
    public native double computeSkinHealth100Jni(double wrinkleScore, double spotsScore, double sensitivityScore, double impuritiesScore, double keratinScore, double poresScore);
    public native String skinQuestionnaire102Jni(String answers);
    public native double computationSkinSensitivity101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
    public native double computationSkinWrinkles101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
    public native double computationSkinOilinessSebum101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
    public native double computationSkinPigmentationSpots101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
    public native double computationSkinCondition101Jni(double moistureScore, double sebumComputationScore);	// deprecated!!! deprecated!!!
    public native String computationSkinConditionCFA100Jni(double mScoreT, double oilinessScore, double mScoreU, double skinConditionQAScore); // For CFA
    public native String computationSkinConditionFfaChp101Jni(double oilinessScore, double skinConditionQAScore); // For FFA, CHP in DBeA.
    public native String computationSkinCondition102Jni(double mScoreT, double sScoreT, double mScoreU, double sScoreU, double QAScore); // 2023-10-10 for CNDP Skin, CMA Skin
    
	// -- (1.1) -- Computation for CNDP Skin only. To calculate average scores of multiple images.
    public native double computationCNDPSkinShine100Jni(double[] analysisResults, int resultCount);
    public native double computationCNDPSkinImpurities100Jni(double[] analysisResults, int resultCount);
    public native double computationCNDPSkinKeratin100Jni(double[] analysisResults, int resultCount);
    public native double computationCNDPSkinPores100Jni(double[] analysisResults, int resultCount);
    
    // -- (2.0) -- Commonly used by CNDP Hair and HH.
    public native String hairQuestionnaire106Jni(String answers);
    public native double computationSclapRedness101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
    public native double computationScalpKeratin101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
    public native double computationHairDensity100Jni(double[] analysisResults, int resultCount);
    public native double computationHairThickness101Jni(double[] analysisResults, int resultCount);
	public native double computationHairSebum101Jni(double[] analysisResults, double questionnaireScore, int resultCount);
	public native double computationScalpCondition100Jni(double moistureScore, double sebumComputationScore);
	public native String computationScalpCondition101Jni(double mScore, double sScore, double QAScore);		// 2023-10-10
	public native double computationHairHealth100Jni(double densityScore, double lossScore, double keratinScore, double rednessScore, double sebumOilinessScore);
	
    // -- (2.1) -- Computation for CNDP hair only.
    public native double computationCNDPHairLoss103Jni(double score1, double score2, double score3, double score4, double QAscore);	// 2023-04-05, updated by Shu Li.
    
	// -- (2.2) -- Computation for CMA-Hair.
    public native String questionnaireCMAHair103Jni(String answers);
	
	// -- (2.3) -- Computation for HH
    public native double computationHHHairLoss102Jni(double[] analysisResults, double questionnaireScore, int resultCount);
	
	// ----------- (3) ----------- Computation for CMA1-Skin.
	public native String questionnaireCMA1Skin102Jni(String answers);
}
