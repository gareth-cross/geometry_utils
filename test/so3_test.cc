#include <chrono>
#include <iostream>

#include "numerical_derivative.hpp"
#include "so3.hpp"
#include "test_utils.hpp"

// TODO(gareth): I think gtest has some tools for testing both double/float.
// Should probably use those here instead of calling manaually.
namespace math {

// A bunch of random vectors w/ norms in the range [0, 2pi].
// Generated offline for use in the tests.
static const std::vector<Eigen::Vector3d> kRandomVectors = {
    {7.83178501e-01, 2.27101030e+00, -3.01517066e+00},
    {-8.76651215e-02, 3.31488940e+00, 2.01268977e+00},
    {-1.52505785e+00, 1.38073327e+00, 5.70658716e-02},
    {2.27092552e+00, 2.10078938e+00, -1.32209595e+00},
    {3.99270084e+00, -2.29989267e+00, -2.45588003e+00},
    {1.51139938e-01, 1.05129036e+00, 2.29256701e+00},
    {1.38766063e-01, -1.73046472e+00, -3.10209909e+00},
    {2.41305679e+00, 2.08998076e+00, -2.06396602e+00},
    {-5.56541962e-01, 5.71259290e-02, 1.56370060e+00},
    {7.45079005e-01, -4.14591148e+00, -1.98107328e-01},
    {-7.79658806e-01, -6.25164776e-01, -5.69857545e-01},
    {-1.14226633e+00, 1.58200471e+00, 6.58515258e-01},
    {-1.34273080e+00, -1.67483502e+00, -1.53829177e+00},
    {1.66657106e+00, 5.84590294e-01, 5.52991474e+00},
    {1.50425882e+00, 1.94414419e+00, -3.07313458e+00},
    {-5.73787869e-01, 8.35724227e-02, -5.92210324e-01},
    {8.57974973e-02, 7.21087229e-02, -7.64704311e-02},
    {2.57481746e+00, 4.55591120e-01, 3.84972194e+00},
    {-1.43525623e+00, -1.51364584e+00, -1.22934938e+00},
    {1.52307473e+00, 2.52262615e-01, -1.33978646e+00},
    {-2.25097561e-01, 2.24669561e-01, 6.39677267e-01},
    {-1.98374649e+00, 2.60985478e+00, 1.28046291e+00},
    {2.70197441e+00, 2.62460784e+00, -2.08983971e-01},
    {1.87914213e-01, 2.17026936e-01, 1.26535291e-01},
    {2.64924186e-01, -7.03289365e-02, 5.61799424e-02},
    {-3.77738563e+00, 1.71866706e+00, 2.14918966e+00},
    {-1.19914921e+00, -1.07215145e+00, -2.61432202e-01},
    {-1.15399357e+00, 3.95648495e+00, 2.93134894e+00},
    {-8.41920018e-01, 2.32755428e+00, -5.10030727e-02},
    {4.36124741e-01, -1.97230163e+00, 2.23220461e+00},
    {-6.73465159e-03, -2.93247402e-02, -4.51756873e-02},
    {2.44263399e+00, -2.96050129e+00, -3.13478824e+00},
    {-6.81581209e-01, -1.33913020e+00, -8.02650731e-01},
    {2.10789301e+00, 6.44350184e-01, 3.67418344e+00},
    {4.58816425e-01, 2.74028404e+00, 1.82392786e+00},
    {2.04593605e+00, -3.27519508e+00, -1.99393915e+00},
    {-1.33196729e-01, 1.71608398e-01, -2.99350662e-02},
    {-1.03253132e+00, -1.38494637e+00, 2.79783319e+00},
    {3.22693003e+00, -2.38235155e+00, 1.14448430e+00},
    {2.93775108e+00, 3.47851042e+00, 2.38665823e+00},
    {9.48686252e-01, 4.88024345e+00, -1.27157875e+00},
    {9.08430960e-01, -2.42596141e+00, 1.96759704e+00},
    {5.26885596e-01, 1.51660670e+00, -1.45345810e+00},
    {-3.91657769e-01, -2.28862474e-01, 1.42604659e+00},
    {1.01165543e+00, 8.67495285e-02, 2.21475212e+00},
    {1.97347254e+00, -1.40603292e+00, 8.37564535e-01},
    {1.44387900e-01, 3.01182801e-01, 2.29081621e-01},
    {5.53838795e-01, -2.82792419e+00, 2.73499458e+00},
    {-7.55153178e-01, 2.72575636e+00, 1.84578034e+00},
    {-1.44264066e-01, 7.51083800e-01, 5.49606183e-01},
    {-3.77236746e+00, 4.65945502e+00, 6.92788031e-01},
    {4.86081572e+00, 1.56583765e+00, 1.47943823e+00},
    {-3.63679638e+00, -4.03267270e+00, 2.84767389e+00},
    {-7.49183075e-01, -7.33134709e-01, 4.51639618e-01},
    {1.32488778e+00, 1.04192363e+00, -5.47372312e-01},
    {7.88813095e-01, 2.86298351e+00, 4.19330646e+00},
    {2.61839218e+00, -3.98132217e-01, -7.43538154e-01},
    {-2.18333463e+00, -1.93288574e+00, -1.14200692e+00},
    {-1.82663117e+00, -1.17212277e+00, 2.38316806e+00},
    {-1.19362573e+00, -1.92179356e+00, -7.56835331e-01},
    {-1.24832753e+00, -2.22610476e+00, -3.15364491e-01},
    {3.45142990e-01, 8.01314836e-01, 4.89966446e-01},
    {3.07556251e+00, 9.25185658e-01, 2.48151092e+00},
    {1.24341410e+00, 1.41916710e+00, 3.63957082e-01},
    {-2.27425353e+00, 1.88278700e-01, -1.40881983e+00},
    {1.82976652e+00, -7.99515873e-01, 5.52742755e-01},
    {-3.53021453e-01, -6.25657513e-02, 3.88925943e-01},
    {-2.48507999e+00, -2.61212215e-01, 2.46279073e+00},
    {2.77192839e-01, 2.73648432e-01, -1.13733822e-01},
    {2.48999029e-02, -5.14408927e-02, -4.03066758e-02},
    {7.61114196e-01, 2.43559407e+00, -3.15868565e-01},
    {-1.44913066e+00, 5.15502264e+00, -2.44562463e+00},
    {4.99919636e+00, 2.52012141e-01, 2.83165268e+00},
    {9.15656481e-01, -2.91345684e+00, 1.27267667e+00},
    {6.33317358e-02, 8.48356136e-02, -1.47900397e-02},
    {-2.79125136e-01, 1.60702169e-01, -7.96463393e-01},
    {-3.61352102e+00, -4.03229701e+00, 6.00273961e-01},
    {5.24892300e+00, 1.10745735e+00, -1.96225610e+00},
    {-1.28441616e+00, -5.85962159e-01, 9.89862342e-01},
    {6.22136579e-02, 3.11781459e-01, 4.08145723e-01},
    {-1.67813767e+00, 3.35789936e-01, 3.57523333e+00},
    {7.54449013e-01, 1.21601425e+00, 3.27310648e+00},
    {-3.93041859e-01, -2.65032804e+00, 2.79158087e+00},
    {1.97828382e+00, 1.93856584e+00, -1.55854678e+00},
    {1.59037873e+00, 2.71292150e+00, 3.30390560e+00},
    {2.38231865e+00, -1.93310801e+00, 7.30951342e-01},
    {-5.34901298e-01, -5.67417590e+00, 1.64924480e+00},
    {-2.79709528e+00, -1.36785060e+00, -5.21285476e+00},
    {4.68221923e+00, -7.28007317e-01, -3.17241871e+00},
    {2.74477626e+00, -1.20987918e+00, -3.64430501e+00},
    {5.00106148e+00, -2.73668533e+00, 4.71598692e-01},
    {1.18101524e+00, 4.09300207e+00, 3.45504241e-02},
    {4.21437149e+00, 2.56039453e+00, -2.13673014e+00},
    {1.98621699e+00, 2.79627591e+00, 1.42831739e+00},
    {3.75107663e+00, -3.06958829e+00, -2.88952368e+00},
    {4.95225814e-01, 9.87337430e-02, -5.26221510e-01},
    {-4.67588993e+00, 2.26061444e-01, 1.55799412e+00},
    {3.80341114e-01, -2.60762726e-02, 2.42218078e-01},
    {6.38244305e-01, -1.63798871e+00, 1.46139666e+00},
    {6.25939270e-01, 5.33054590e+00, -3.95977050e-01},
    {-1.09339830e+00, -1.42607697e+00, -9.54234191e-01},
    {2.65968726e-01, 1.83689647e-01, 3.85460489e-01},
    {-4.34378655e+00, 4.70460599e-01, 3.00539389e-01},
    {2.28828176e-01, -2.34670282e-01, 1.75454561e-01},
    {3.61974328e-01, -1.72002151e+00, -4.56419065e+00},
    {-1.92947655e+00, 1.09326397e+00, -1.77740190e+00},
    {-2.02971570e+00, -1.20198638e+00, 2.34205301e+00},
    {-1.93334201e+00, 1.15519406e+00, -1.09374363e+00},
    {-1.64259732e+00, -2.91677345e+00, 4.95088702e-02},
    {4.30223711e-01, -2.87527873e-01, 2.57306684e+00},
    {-2.22213282e+00, 2.72562386e-01, -7.48623949e-01},
    {-2.19415876e+00, -1.49414376e+00, -2.33286305e+00},
    {2.92994941e+00, -4.77851360e+00, -5.55483863e-01},
    {-7.09559170e-01, 6.18481188e-01, -1.30154053e+00},
    {7.43976286e-02, -1.26852459e+00, -1.21970519e+00},
    {1.58329790e+00, -2.20900778e+00, 3.51267586e+00},
    {-2.21133863e+00, -4.99223544e+00, 2.28186734e+00},
    {1.07174750e+00, 8.60364588e-01, 1.06645026e+00},
    {1.48522650e+00, -1.91349070e+00, 1.25117736e+00},
    {-1.57271187e+00, 1.53948220e+00, -5.48982547e+00},
    {7.10624446e-01, 4.42897533e-01, 1.51072761e+00},
    {-3.87099763e-01, 3.78263442e-01, -3.97941284e-01},
    {2.51844929e-01, 4.21697222e+00, -4.36896740e-01},
    {2.34071289e+00, -2.24686688e+00, -2.57836171e+00},
    {-8.25667672e-01, 6.73056069e-01, 6.84161289e-01},
    {-1.94167437e-01, -6.52160871e-02, -2.95521345e-01},
    {-2.81978898e+00, -4.54503517e+00, -3.29462877e+00},
    {-1.47071351e+00, 1.02727437e+00, -6.85937463e-02},
    {6.98166058e-01, 1.01493404e+00, -1.07970207e+00},
    {3.21062354e-01, -6.21498867e-01, -6.86578241e-01},
    {-4.14186329e+00, -3.60070557e+00, 2.88931857e+00},
    {-4.99502366e-01, 3.39053528e-01, -6.16956820e-01},
    {-1.92950681e+00, 2.52649200e+00, 1.33850950e+00},
    {1.14650844e+00, -6.28621431e-01, -1.88213660e+00},
    {-3.07618866e-01, 2.55216152e+00, -1.21688279e+00},
    {-4.40251903e+00, -4.08093789e+00, 1.30047662e+00},
    {-4.34517934e-01, 3.05194301e-01, 5.02581484e-01},
    {2.85730706e+00, -5.25330052e+00, 1.60565740e+00},
    {-3.08331031e+00, 3.31716650e+00, -2.02488128e+00},
    {2.19097621e-01, -3.48399518e-01, 3.95453575e-01},
    {2.89588215e+00, -1.91398607e+00, 1.94923736e+00},
    {1.28749065e+00, 1.55018696e-01, 2.93666553e-01},
    {2.61413444e-02, -5.61358241e-02, -8.87320974e-02},
    {-1.31639970e-01, -3.92420739e+00, 3.59608394e+00},
    {1.03026495e+00, 2.50184955e+00, -2.42461017e-01},
    {6.10761858e-02, -7.46895610e-03, -2.50620094e-02},
    {-2.54866683e-01, -4.41562040e-01, 5.49670268e-01},
    {-1.37789002e-01, -4.50968931e-01, 3.15948462e-01},
    {-2.84778071e+00, -2.85524064e+00, 3.44903269e+00},
    {-4.47140340e+00, 4.19660131e+00, 4.54670515e-01},
    {-3.16468254e-01, -3.78266234e-01, -9.68363205e-01},
    {-4.92888597e-01, -1.30627304e-01, 1.01179031e+00},
    {-4.22584469e+00, 4.02059906e+00, -2.24625391e+00},
    {-9.31113456e-01, -1.37113358e+00, 1.05024426e+00},
    {-1.24684847e+00, -1.45771235e+00, -6.03387444e-01},
    {9.83219968e-01, -5.35413079e-01, 3.69681414e+00},
    {-2.27664037e+00, 2.97339028e+00, -6.39508794e-01},
    {3.05382433e-01, 3.97783137e+00, -4.50264523e+00},
    {7.75264205e-01, -8.90039757e-01, -1.46463232e+00},
    {5.33527445e-03, -4.78015639e-01, 5.44650918e-01},
    {-1.74834936e+00, 2.93197292e+00, -3.62554117e+00},
    {-2.24103443e+00, 2.40117558e+00, -2.30739313e-01},
    {-5.41655701e-01, 5.04923027e-01, 3.25279419e-01},
    {-1.63302771e+00, -1.95884070e+00, -2.21791774e+00},
    {8.33311308e-01, 8.49810150e-01, 1.97402471e+00},
    {2.83375457e+00, -2.87398588e+00, -9.76297695e-01},
    {4.12431438e+00, 7.96655136e-01, 1.65318916e-01},
    {-4.27575748e+00, -2.80047758e+00, 3.10397340e+00},
    {-2.74148710e+00, -1.41678581e+00, 1.87031251e+00},
    {2.50988929e+00, 3.19843059e+00, 1.78061972e+00},
    {-6.19726190e-02, 9.66639857e-01, -2.86360221e-01},
    {5.80003402e-01, 1.12659269e+00, -5.17200741e-01},
    {4.02646411e+00, -1.81547713e+00, -1.16931099e+00},
    {-1.16544735e+00, -2.73748890e+00, -3.07532384e+00},
    {-1.35972024e+00, 3.60455073e+00, -3.06699407e+00},
    {5.80354549e-01, 4.24426945e+00, 1.83596530e+00},
    {4.96955974e+00, -2.45296514e+00, -3.38323135e-01},
    {-5.07823088e+00, -1.22872128e+00, 9.46033354e-01},
    {5.94763278e-01, 1.01697301e+00, 8.93062660e-01},
    {1.62559834e+00, 2.95342115e+00, -1.57922166e+00},
    {-2.92800959e+00, 2.50171445e+00, 2.49607624e+00},
    {6.14894711e-01, 6.63417338e-01, 2.62731832e-01},
    {-2.35199803e-01, -2.22036451e-01, -5.91655518e+00},
    {-6.52896303e-01, 7.03004511e-01, 3.99978954e-01},
    {-1.17842643e+00, -1.66573326e-01, 1.74363266e+00},
    {3.21631425e+00, -2.59652108e+00, 2.04513052e+00},
    {6.39311059e-01, 2.80195995e+00, 3.10974851e+00},
    {-6.78007478e-01, 3.67886766e+00, 1.29607874e+00},
    {5.31737383e+00, 2.35150511e+00, 5.38269332e-01},
    {-1.75811660e+00, 1.43197360e+00, -5.01471550e-01},
    {1.07778991e-01, -1.39572210e+00, -1.38142973e+00},
    {-8.79033633e-01, 1.15354895e+00, 1.08987269e+00},
    {-8.72544155e-01, 1.45269953e+00, -2.83234679e+00},
    {-1.73940924e+00, -1.64468402e+00, -5.70127846e-01},
    {3.00967629e+00, 4.26308272e-01, -1.19755198e+00},
    {-3.69534023e+00, 2.08581955e+00, -3.61611154e-02},
    {-4.81237979e-01, -6.44116434e-01, -9.71845118e-01},
    {-4.80298242e-01, 5.32533532e-01, 2.39900406e-01},
    {1.89358855e-01, -1.55717846e+00, -4.82748643e+00},
    {-2.21342736e+00, -8.32295810e-01, -1.35061886e-01},
    {2.40053464e+00, 9.54020733e-03, -5.37054939e+00},
    {-1.19673888e-01, 6.14927685e-02, 1.82267088e-01},
    {-7.96271337e-01, -2.20354075e+00, -1.90436131e+00},
    {-8.03401699e-02, -3.79942270e+00, 3.59163557e+00},
    {4.63736900e-01, 1.97329948e-01, 1.82524403e+00},
    {4.33200570e-01, 2.55965068e+00, 9.52449873e-01},
    {-8.93822520e-01, -5.98727704e-01, 1.07616092e+00},
    {-2.32754165e-02, 1.11889280e-01, -2.16063184e-01},
    {-4.03134592e+00, -2.17744484e+00, 8.31654559e-01},
    {-1.23369414e+00, -1.08912429e+00, 1.10173468e+00},
    {-2.28859094e-01, 3.71804821e-01, -2.09816765e+00},
    {3.68760776e-01, 1.70584583e+00, -2.45195795e-01},
    {2.25989631e+00, -5.06973104e-02, -2.26911234e+00},
    {8.02628765e-01, 3.87033652e+00, 4.87758062e+00},
    {3.83732709e+00, 3.71807235e+00, -1.47288887e+00},
    {-3.59246007e-01, -3.38750159e+00, 3.34817193e+00},
    {-6.02372386e-01, 5.45179892e-01, 4.82988095e-02},
    {1.31528760e-01, -9.27671776e-01, 7.30371312e-01},
    {-7.92568832e-02, -5.67251686e-03, -2.81422741e-03},
    {-7.08963446e-01, -2.07024440e+00, 2.00355657e-01},
    {3.20695486e+00, -1.56848614e+00, 4.27390598e+00},
    {-2.20547496e+00, -1.24689870e+00, 1.23868810e+00},
    {-1.67106211e-01, -9.31881709e-01, -4.71240299e-01},
    {1.05177152e-01, -4.88395920e+00, -2.57905842e+00},
    {-1.46519342e+00, -7.81909133e-01, 5.36944354e-01},
    {1.27662499e+00, -5.49447189e+00, 1.10571499e-01},
    {1.64959183e+00, 3.64509208e+00, -3.62634261e+00},
    {-1.47646727e+00, -3.34002305e+00, 3.75159240e+00},
    {2.13441607e-01, -1.89545437e+00, 1.99713412e+00},
    {-1.25230247e+00, -3.94556621e+00, -4.38642994e+00},
    {-2.72451176e+00, -2.44431418e+00, 2.99490469e+00},
    {2.65590629e+00, -3.86140810e+00, 3.52121740e+00},
    {1.43347503e-02, 4.16507434e-01, 1.26354554e-01},
    {9.56445836e-01, 2.02281984e+00, 5.74647183e-02},
    {2.01526365e+00, -4.36416248e+00, 3.65637235e+00},
    {-2.46246193e+00, 2.46696511e+00, -2.77356695e+00},
    {-2.65927755e+00, 1.33776519e+00, -4.40922690e+00},
    {-1.76107340e+00, 2.40502617e+00, 4.41812391e-01},
    {1.68348305e+00, -1.42043703e+00, -1.61108340e+00},
    {3.29063480e+00, 1.31977114e+00, 1.26465266e+00},
    {-3.20929404e-02, -1.54277067e+00, -3.96942085e+00},
    {3.74555026e-01, -3.72814822e+00, -2.43878183e+00},
    {-2.18207644e-01, 2.09904246e+00, -2.60936793e+00},
    {1.45870699e+00, -1.58428697e-01, 2.50753204e+00},
    {-7.37104946e-01, -1.35466040e+00, 1.73497863e+00},
    {-1.53330019e-01, -9.72172865e-01, 7.44171394e-02},
    {-2.79937076e-01, 3.21590088e-01, 7.06121768e-01},
    {-5.29227940e-01, -6.88903016e-01, 1.51716413e+00},
    {-1.78045383e-01, 1.10822425e-01, -9.63727516e-02},
    {-1.95975126e-01, -2.31594195e+00, 1.90776019e+00},
    {1.39426798e+00, 1.62957278e+00, 1.47536519e+00},
    {1.68565832e+00, -8.28015954e-02, 1.67579779e+00},
    {-2.37595509e-01, 5.61763236e-01, -2.93297266e+00},
    {-3.35904633e+00, 7.69759401e-01, -1.71797831e+00},
    {2.17411332e+00, 4.16760307e+00, 2.91420194e+00},
    {3.85232664e+00, 1.82300266e+00, -6.99245104e-01},
    {4.22108508e-01, -9.10672807e-01, -1.56623003e-01},
    {-2.32397464e+00, -4.14964455e+00, -2.69371238e+00},
    {7.04435083e-01, -6.70670609e-01, 5.80655771e-02},
    {1.06451624e+00, 7.21645834e-01, -1.22974981e-01},
    {2.79864544e+00, 2.21375670e+00, -3.22219918e+00},
    {7.09984639e-01, -6.07989629e-01, -1.87594867e-01},
    {1.03725718e+00, 2.25310661e+00, -1.22772024e+00},
    {1.38758798e-01, -4.43359922e-01, -3.13805366e-01},
    {-1.11624450e+00, 1.82543406e+00, 2.63699224e+00},
    {-1.21166771e+00, -1.49721507e+00, -2.29006067e-01},
    {-2.80806868e+00, 1.73171033e+00, 2.96549307e+00},
    {1.64518867e+00, -7.01968047e-01, 8.09509929e-01},
    {2.93024241e+00, 6.00097265e-01, 1.76661265e+00},
    {2.34225167e+00, -2.28699861e+00, -2.09213117e+00},
    {-5.35679547e-01, 2.53112508e-01, -3.55454755e-01},
    {8.05725236e-02, -2.83457368e-01, 2.13410899e-01},
    {-1.52741739e+00, -5.38129339e-01, 5.46828825e+00},
    {7.98990314e-01, -7.86438474e-01, 8.11783667e-01},
    {2.53848534e+00, 2.50188633e+00, 9.84639609e-01},
    {-2.23054840e+00, -1.30208303e+00, 3.98085605e+00},
    {-2.20984527e+00, 5.23749347e+00, -1.54317889e+00},
    {4.56416733e-01, -1.85633746e-01, 3.01630441e-02},
    {-2.92648977e-01, 3.60965155e+00, 4.67744584e-01},
    {2.06236179e+00, -1.37383852e+00, 2.51201584e-01},
    {2.54780136e+00, 2.38807277e+00, -6.89549045e-01},
    {-1.58530037e-01, 2.14224048e-01, -1.79144859e-01},
    {-2.01345578e+00, 4.37298697e+00, 1.44011860e+00},
    {-1.71636053e-01, -7.99083655e-01, -2.76634553e+00},
    {2.56382952e-01, 1.48389038e+00, 1.38531732e+00},
    {-1.83848503e+00, -2.17183110e+00, -1.40683263e+00},
    {2.77355009e+00, 1.36757363e+00, -8.21564035e-01},
    {-1.60893045e+00, -1.72998741e+00, 6.63758483e-01},
    {3.51573357e-02, -6.51843395e-01, -1.69657471e+00},
    {2.34077120e+00, 4.17708332e+00, -2.41715052e+00},
    {2.48043823e-01, 3.15844507e-02, 2.40807556e-01},
    {-3.07497463e+00, -3.98670637e+00, -2.51264349e+00},
    {1.79002410e+00, -1.73723679e+00, 8.11365390e-01},
    {-1.10910677e+00, -2.97719745e+00, 2.94569114e+00},
    {-1.33893368e+00, -6.12049880e-01, 2.04079296e+00},
    {4.05592648e+00, -1.61141724e+00, -2.84083439e+00},
    {1.91897135e+00, -9.84299391e-01, -8.16851356e-01},
    {-4.81716723e-01, 4.98611737e-01, 6.87655724e-01},
    {2.11820469e+00, 1.07971334e+00, 2.97438359e+00},
    {-1.24966361e+00, 1.89205885e-01, 3.85406085e+00}};

// Test skew-symmetric operator.
TEST(SO3Test, TestSkew3) {
  using Vector3d = Vector<double, 3>;
  using Matrix3d = Matrix<double, 3, 3>;
  const Vector3d x = (Vector3d() << 1, 2, 3).finished();
  const Vector3d y = (Vector3d() << 1, 1, 1).finished();
  EXPECT_EIGEN_NEAR(Vector3d::Zero(), Skew3(x) * x, tol::kPico);
  EXPECT_EIGEN_NEAR(x.cross(y), Skew3(x) * y, tol::kPico);
  EXPECT_EIGEN_NEAR(Matrix3d::Zero(), Skew3(x) + Skew3(x).transpose(), tol::kPico);
}

// Test quaternion multiplication.
TEST(SO3Test, TestQuaternionMulMatrix) {
  const auto to_vec = [](const Quaternion<double>& q) -> Vector<double, 4> {
    return Vector<double, 4>(q.w(), q.x(), q.y(), q.z());
  };
  const Quaternion<double> q0{-0.5, 0.2, 0.1, 0.8};
  const Quaternion<double> q1{0.4, -0.3, 0.2, 0.45};
  EXPECT_EIGEN_NEAR(to_vec(q0 * q1), QuaternionMulMatrix(q0) * to_vec(q1), tol::kPico);
  EXPECT_EIGEN_NEAR(to_vec(q1 * q0), QuaternionMulMatrix(q1) * to_vec(q0), tol::kPico);
}

// Simple test of exponential map by series comparison + numerical derivative.
class TestQuaternionExp : public ::testing::Test {
 public:
  template <typename Scalar>
  void TestOmega(const Vector<Scalar, 3>& w, const Scalar matrix_tol,
                 const Scalar deriv_tol) const {
    // check that the derivative and non-derivative versions are the same
    const Quaternion<Scalar> just_q = math::QuaternionExp(w);
    const QuaternionExpDerivative<Scalar> q_and_deriv{w};
    ASSERT_EIGEN_NEAR(just_q.matrix(), q_and_deriv.q.matrix(), tol::kPico);

    // compare to the exponential map as a power series ~ 50 terms
    ASSERT_EIGEN_NEAR(ExpMatrixSeries(Skew3(w), 50), q_and_deriv.q.matrix(), matrix_tol)
        << "w = " << w.transpose();

    // compare to Eigen implementation for good measure
    const Eigen::AngleAxis<Scalar> aa(w.norm(), w.normalized());
    ASSERT_EIGEN_NEAR(aa.toRotationMatrix(), q_and_deriv.q.matrix(), matrix_tol)
        << "w = " << w.transpose();

    // check derivative numerically
    const Matrix<Scalar, 4, 3> J_numerical =
        NumericalJacobian(w, [](const Vector<Scalar, 3>& w) -> Vector<Scalar, 4> {
          // convert to correct order here
          const Quaternion<Scalar> q = math::QuaternionExp(w);
          return Vector<Scalar, 4>(q.w(), q.x(), q.y(), q.z());
        });
    ASSERT_EIGEN_NEAR(J_numerical, q_and_deriv.q_D_w, deriv_tol) << "w = " << w.transpose();
  }

  void Test() const {
    // vec3d is not aligned (not 16 byte multiple)
    // clang-format off
    const std::vector<Eigen::Vector3d> vectors = {
      {-M_PI, 0, 0},
      {0, M_PI, 0},
      {0, 0, -M_PI},
    };
    // clang-format on
    // test near pi
    for (const Eigen::Vector3d& w : vectors) {
      TestOmega<double>(w, tol::kPico, tol::kNano);
      TestOmega<float>(w.cast<float>(), tol::kMicro, tol::kMilli / 10);
    }
    for (const Eigen::Vector3d& w : kRandomVectors) {
      TestOmega<double>(w, tol::kPico, tol::kNano);
      TestOmega<float>(w.cast<float>(), tol::kMicro, tol::kMilli / 10);
    }
  }

  void TestNearZero() const {
    TestOmega<double>({1.0e-7, 0.5e-6, 3.5e-8}, tol::kNano, tol::kMicro);
    TestOmega<double>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
    TestOmega<float>({1.0e-7, 0.5e-6, 3.5e-8}, tol::kNano, tol::kMicro);
    TestOmega<float>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
  }
};

TEST_FIXTURE(TestQuaternionExp, Test);
TEST_FIXTURE(TestQuaternionExp, TestNearZero);

// Check that RotationLog does the inverse of QuaternionExp.
TEST(SO3Test, TestRotationLog) {
  // test quaternion
  const Vector<double, 3> v1{-0.7, 0.23, 0.4};
  const Quaternion<double> r1 = QuaternionExp(v1);
  EXPECT_EIGEN_NEAR(v1, RotationLog(r1), tol::kPico);
  // test matrix
  const Vector<float, 3> v2{0.01, -0.5, 0.03};
  const Quaternion<float> r2 = QuaternionExp(v2);
  EXPECT_EIGEN_NEAR(v2, RotationLog(r2.matrix()), tol::kMicro);
  // test identity
  const auto zero = Vector<double, 3>::Zero();
  EXPECT_EIGEN_NEAR(zero, RotationLog(Quaternion<double>::Identity()), tol::kPico);
  EXPECT_EIGEN_NEAR(zero.cast<float>(), RotationLog(Matrix<float, 3, 3>::Identity()), tol::kPico);
  // test some randomly sampled vectors
  for (const Eigen::Vector3d& w : kRandomVectors) {
    const Eigen::Matrix3d R = QuaternionExp(w).matrix();
    // make sure it's the same rotation we get back out
    EXPECT_EIGEN_NEAR(R, QuaternionExp(RotationLog(R)).matrix(), tol::kNano);
    EXPECT_EIGEN_NEAR(R.cast<float>(), QuaternionExp(RotationLog(R.cast<float>())).matrix(),
                      tol::kMicro);
  }
}

// Test the SO(3) jacobian.
class TestSO3Jacobian : public ::testing::Test {
 public:
  template <typename Scalar>
  static void TestJacobian(const Vector<Scalar, 3>& w_a, const Scalar deriv_tol) {
    const Matrix<Scalar, 3, 3> J_analytical = math::SO3Jacobian(w_a);
    // This jacobian is only valid for small `w`, so evaluate about zero.
    const Matrix<Scalar, 3, 3> J_numerical =
        NumericalJacobian(Vector<Scalar, 3>::Zero(),
                          [&](const Vector<Scalar, 3>& w) { return QuaternionExp(w_a + w); });
    EXPECT_EIGEN_NEAR(J_numerical, J_analytical, deriv_tol);

    PRINT(J_analytical);
    PRINT(J_numerical);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomVectors) {
      PRINT(w.transpose());
      TestJacobian<double>(w, tol::kNano);
      break;
//      TestJacobian<float>(w.cast<float>(), tol::kNano);
    }
  }

  void TestNearZero() {}
};

TEST_FIXTURE(TestSO3Jacobian, TestGeneral);
TEST_FIXTURE(TestSO3Jacobian, TestNearZero);

// Test the derivative of the exponential map, matrix form.
class TestMatrixExpDerivative : public ::testing::Test {
 public:
  template <typename Scalar>
  static Vector<Scalar, 9> VecExpMatrix(const Vector<Scalar, 3>& w) {
    // Convert to vectorized format.
    const Matrix<Scalar, 3, 3> R = math::QuaternionExp(w).matrix();
    return Eigen::Map<const Vector<Scalar, 9>>(R.data());
  }

  template <typename Scalar>
  static void TestDerivative(const Vector<Scalar, 3>& w, const Scalar deriv_tol) {
    const Matrix<Scalar, 9, 3> D_w = math::SO3ExpMatrixDerivative(w);
    const Matrix<Scalar, 9, 3> J_numerical =
        NumericalJacobian(w, &TestMatrixExpDerivative::VecExpMatrix<Scalar>);
    ASSERT_EIGEN_NEAR(J_numerical, D_w, deriv_tol);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomVectors) {
      TestDerivative<double>(w, tol::kNano / 10);
      TestDerivative<float>(w.cast<float>(), tol::kMilli / 10);
    }
  }

  void TestNearZero() {
    TestDerivative<double>({-1.0e-7, 1.0e-8, 0.5e-6}, tol::kMicro);
    TestDerivative<float>({-1.0e-7, 1.0e-8, 0.5e-6}, tol::kMicro);

    // at exactly zero it should be identically equal to the generators of SO(3)
    const Matrix<double, 9, 3> J_at_zero =
        math::SO3ExpMatrixDerivative(Vector<double, 3>::Zero().eval());
    const auto i_hat = Vector<double, 3>::UnitX();
    const auto j_hat = Vector<double, 3>::UnitY();
    const auto k_hat = Vector<double, 3>::UnitZ();
    EXPECT_EIGEN_NEAR(Skew3(-i_hat), J_at_zero.block(0, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(Skew3(-j_hat), J_at_zero.block(3, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(Skew3(-k_hat), J_at_zero.block(6, 0, 3, 3), tol::kPico);
  }
};

TEST_FIXTURE(TestMatrixExpDerivative, TestGeneral);
TEST_FIXTURE(TestMatrixExpDerivative, TestNearZero);

// Have to be careful when testing this method numerically, since the output of log() can
// jump around if the rotation R * exp(w) is large.
TEST(SO3Test, SO3LogMulExpDerivative) {
  // create the matrix R we multiply against
  const Vector<double, 3> R_log{0.6, -0.1, 0.4};
  const Quaternion<double> R = math::QuaternionExp(R_log);

  // functor that holds R fixed and multiplies on w
  const auto fix_r_functor = [&](const Vector<double, 3>& w) -> Vector<double, 3> {
    return math::RotationLog(R * math::QuaternionExp(w));
  };

  // try a bunch of values for omega
  // clang-format off
  const std::vector<Vector<double, 3>> samples = {
    {0.6, -0.1, 0.4},
    {0.8, 0.0, 0.2},
    {-1.2, 0.6, 1.5},
    {0.0, 0.0, 0.1},
    {1.5, 1.7, -1.2},
    {-0.3, 0.3, 0.3},
    {M_PI / 2, 0, 0},
    {0, 0, -M_PI / 4},
    {0.5, 0.0, M_PI / 2},
  };
  // clang-format on
  for (const auto& w : samples) {
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpDerivative(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano) << "w = " << w.transpose();
  }
}

TEST(SO3Test, SO3LogMulExpDerivativeNearZero) {
  // test small angle cases
  // clang-format off
  const std::vector<Vector<double, 3>> samples = {
    {0.0, 0.0, 0.0},
    {-1.0e-5, 1.0e-5, 0.3e-5},
    {0.1e-5, 0.0, -0.1e-5},
    {-0.2e-8, 0.3e-7, 0.0},
  };
  // clang-format on

  // for small hangle to hold, R should be identity
  const Quaternion<double> R = Quaternion<double>::Identity();

  // functor that holds R fixed and multiplies on w
  const auto fix_r_functor = [&](const Vector<double, 3>& w) -> Vector<double, 3> {
    return math::RotationLog(R * math::QuaternionExp(w));
  };

  for (const auto& w : samples) {
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpDerivative(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano);
  }
}

}  // namespace math
