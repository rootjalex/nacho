#pragma once

namespace nacho {

struct Level;
struct Format;
struct TensorType;

struct Expr;
struct Add;
struct Bc;
struct Mul;
struct Sum;
struct Tensor;

struct Seq;
struct Empty;
struct Index;
struct Intersect;
struct Union;
struct Universe;

struct cExpr;
struct cAdd;
struct cMul;
struct cTensor;

struct CIN;
struct Accumulate;
struct Assign;
struct Forall;
struct Sequence;
struct Where;
struct CalculateWork;

namespace llir {

struct lType;
struct Generic_t;
struct Int_t;
struct Float_t;
struct Ptr_t;
struct Tuple_t;
struct Struct_t;

struct lExpr;
struct lOp;
struct lBinOp;
struct lConst;
struct lBuild;
struct lSelect;
struct lArrayAccess;
struct lFieldAccess;
struct lPtrAccess;
struct lVar;
struct lFunctionCall;
struct lIncrement;
struct lAddress;

struct lStmt;
struct Declare;
struct IfElse;
struct Return;
struct Sequence;
struct Store;
struct While;
struct For;
struct Function;
struct BaseExpr;
struct Break;
struct Accumulate;
struct KernelLaunch;
struct RawCode;

} // namespace llir

} // namespace nacho
